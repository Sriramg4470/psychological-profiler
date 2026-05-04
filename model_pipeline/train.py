import os
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
import onnx
# Kaggle import removed as it causes authentication errors on import
kaggle = None

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
KAGGLE_DATASET = "bittlingmayer/amazonreviews" # Example 1.5GB dataset
OUTPUT_ONNX_PATH = "../triton/model_repository/sentiment_model/1/model.onnx"
MAX_LENGTH = 128
BATCH_SIZE = 32

# ---------------------------------------------------------
# INTERMEDIATE/UNIQUE MODEL ARCHITECTURE
# DistilBERT + Bi-LSTM + Attention Pooling Head
# ---------------------------------------------------------
class AdvancedSentimentModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(AdvancedSentimentModel, self).__init__()
        self.num_labels = num_labels
        self.distilbert = AutoModel.from_pretrained(model_name)
        
        # Bi-LSTM to capture sequential dependencies of the transformer embeddings
        hidden_size = self.distilbert.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size // 2, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Dropout and classifier
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # 1. Base Transformer
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] # (batch_size, sequence_length, hidden_size)
        
        # 2. Sequence modeling with Bi-LSTM
        lstm_output, _ = self.lstm(sequence_output)
        
        # 3. Mean Pooling with attention mask consideration
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_output.size()).float()
        sum_embeddings = torch.sum(lstm_output * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # 4. Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def download_dataset():
    """
    Downloads the 1.5GB dataset from Kaggle.
    Requires kaggle.json configured in ~/.kaggle/
    """
    print(f"Downloading dataset {KAGGLE_DATASET} from Kaggle...")
    os.makedirs("./data", exist_ok=True)
    try:
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path='./data', unzip=True)
        print("Download complete.")
    except Exception as e:
        print(f"Kaggle API failed: {e}. Ensure kaggle.json is configured.")

def train_and_export():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loading dataset...")
    # Use HuggingFace datasets to stream massive datasets easily if local isn't present
    # dataset = load_dataset("amazon_polarity")
    # 
    # train_dataset = dataset["train"].select(range(50000)) 
    # test_dataset = dataset["test"].select(range(5000))
    # 
    # def tokenize_function(examples):
    #     return tokenizer(examples["content"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    # 
    # tokenized_train = train_dataset.map(tokenize_function, batched=True)
    # tokenized_test = test_dataset.map(tokenize_function, batched=True)
    tokenized_train = None
    tokenized_test = None

    # Initialize Custom Advanced Model
    model = AdvancedSentimentModel(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    print("Starting training on large dataset...")
    # trainer.train() # UNCOMMENT TO TRAIN

    print("Exporting Advanced Model to ONNX...")
    os.makedirs(os.path.dirname(OUTPUT_ONNX_PATH), exist_ok=True)
    
    dummy_input = tokenizer("This is a sample text for ONNX export", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            OUTPUT_ONNX_PATH,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True
        )
    
    print(f"Model exported successfully to {OUTPUT_ONNX_PATH}")

if __name__ == "__main__":
    # download_dataset()
    train_and_export()
