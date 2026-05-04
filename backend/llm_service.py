import os
from openai import AsyncOpenAI
import logging
from typing import List

logger = logging.getLogger(__name__)

class ExplanationService:
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY", "mock-key-for-testing")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.is_mock = self.api_key == "mock-key-for-testing"
        
        if not self.is_mock:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def generate_mindset_profile(self, paragraph: str, sentence_data: List[dict], majority: str) -> str:
        """
        Generates a psychological profile/mindset of the author based on the sentiment flow.
        """
        # If no key, provide a robust dynamic mock response
        if self.is_mock:
            # Generate a dynamic mock based on the data
            pos = sum(1 for s in sentence_data if s.sentiment == 'Positive')
            neg = sum(1 for s in sentence_data if s.sentiment == 'Negative')
            
            if pos > neg and neg == 0:
                mindset = "The author is in a highly optimistic and enthusiastic mindset. There are no signs of hesitation or negativity, suggesting a very positive experience."
            elif neg > pos and pos == 0:
                mindset = "The author is displaying a deeply dissatisfied or frustrated mindset. The consistent negative tone indicates clear displeasure."
            elif pos >= neg:
                mindset = "The author has a generally positive outlook but exhibits some mixed feelings or reservations. They are likely weighing pros and cons, but leaning towards optimism."
            else:
                mindset = "The author is primarily critical, though they acknowledge some positive aspects. Their mindset is skeptical and somewhat defensive."
                
            return f"{mindset} (Mock Profile - Add an LLM_API_KEY to enable deep Generative AI profiling)"

        try:
            # Format data for the LLM
            flow_str = " -> ".join([f"[{s.sentiment}]" for s in sentence_data])
            
            system_prompt = (
                "You are an expert psychological profiler and behavioral analyst. "
                "Analyze the provided text and the emotional trajectory (sentence by sentence) to determine the author's mindset. "
                "Output a short, insightful paragraph describing their psychological state, emotional shifts, and underlying attitude. "
                "Do not repeat the text. Just give the mindset analysis."
            )

            user_prompt = f"Text: '{paragraph}'\nEmotional Trajectory: {flow_str}\nMajority Sentiment: {majority}\nProfile the author's mindset:"

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to generate profile: {e}")
            return "Unable to generate psychological profile due to an API error."
