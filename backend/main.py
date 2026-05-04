import os
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging

from llm_service import ExplanationService
from triton_client import TritonInferenceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Paragraph Sentiment Analysis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Connecting to Triton Inference Server (gRPC)...")
triton_url = os.getenv("TRITON_URL", "triton:8001")
triton_client = TritonInferenceClient(url=triton_url)
explanation_service = ExplanationService()

class PredictRequest(BaseModel):
    text: str

class SentenceAnalysis(BaseModel):
    text: str
    sentiment: str
    confidence: float

class PredictResponse(BaseModel):
    majority_sentiment: str
    overall_confidence: float
    mindset_profile: str
    sentences: List[SentenceAnalysis]

def split_into_sentences(text: str) -> List[str]:
    # A simple regex to split text into sentences based on punctuation (. ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.strip()) > 0]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        # 1. Sentence Tokenization
        sentences_text = split_into_sentences(request.text)
        if not sentences_text:
            sentences_text = [request.text] # Fallback if no punctuation

        # 2. Analyze each sentence
        analyzed_sentences = []
        pos_count = 0
        neg_count = 0
        total_confidence = 0.0

        for s_text in sentences_text:
            # Call Triton instead of local pipeline
            sent_label, conf = await triton_client.predict_async(s_text)
            
            analyzed_sentences.append(
                SentenceAnalysis(text=s_text, sentiment=sent_label, confidence=conf)
            )
            
            if sent_label == "Positive":
                pos_count += 1
            else:
                neg_count += 1
                
            total_confidence += conf

        # 3. Aggregate
        total_sentences = len(analyzed_sentences)
        majority_sentiment = "Positive" if pos_count >= neg_count else "Negative"
        overall_confidence = total_confidence / total_sentences if total_sentences > 0 else 0.0
        
        if pos_count == neg_count and total_sentences > 1:
            majority_sentiment = "Mixed/Neutral"

        # 4. Psychological Profiling via LLM
        mindset_profile = await explanation_service.generate_mindset_profile(
            paragraph=request.text,
            sentence_data=analyzed_sentences,
            majority=majority_sentiment
        )

        return PredictResponse(
            majority_sentiment=majority_sentiment,
            overall_confidence=overall_confidence,
            mindset_profile=mindset_profile,
            sentences=analyzed_sentences
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
