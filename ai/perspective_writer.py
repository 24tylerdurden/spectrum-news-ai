import anthropic
from openai import OpenAI
import httpx
from dataclasses import dataclass
from typing import Optional
import json
import logging

from clustering import Cluster
from scraper import RawArticle

logger = logging.getLogger(__name__)


@dataclass
class PerspectiveResult:
    topic: str
    left_headline: str
    left_summary: str
    left_body: str
    left_source_name: str
    left_source_url: str
    right_headline: str
    right_summary: str
    right_body: str
    right_source_name: str
    right_source_url: str


class PerspectiveWriter:
    def __init__(self, provider: str = "anthropic", model: str = "claude-sonnet-4-20250514", 
                 api_key: str = None, openai_api_key: str = None, ollama_base_url: str = "http://localhost:11434"):
        self.provider = provider
        self.model = model
        
        if provider == "anthropic":
            if not api_key:
                raise ValueError("Anthropic API key is required for anthropic provider")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for openai provider")
            self.client = OpenAI(api_key=openai_api_key)
        elif provider == "ollama":
            self.ollama_base_url = ollama_base_url
            self.client = httpx.Client(timeout=120.0)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_perspectives(self, cluster: Cluster, left_article_text: str, right_article_text: str) -> Optional[PerspectiveResult]:
        """Generate left and right perspectives using the configured AI provider."""
        if not cluster.left_articles or not cluster.right_articles:
            logger.error("Cluster must have at least one left and one right article")
            return None
        
        left_article = cluster.left_articles[0]
        right_article = cluster.right_articles[0]
        
        system_prompt = """You are a senior political journalist writing for an Indian news aggregator called "Biased India". Your job is to present the same news story from two distinct political perspectives based on how real Indian media outlets actually covered it."""
        
        user_prompt = f"""Below are real articles from Indian news outlets covering the same story.

LEFT-LEANING SOURCE ({left_article.source_name}):
Headline: {left_article.title}
Content: {left_article_text}

RIGHT-LEANING SOURCE ({right_article.source_name}):
Headline: {right_article.title}
Content: {right_article_text}

Generate a structured JSON response with exactly this shape:
{{
  "topic": "neutral 6-10 word headline summarizing the story",
  "left": {{
    "headline": "headline written in the framing of the left source",
    "summary": "2-3 sentence summary of how left media covered this",
    "body": "HTML string with 3-4 <p> tags OR a <ul>/<ol> list. Reflect the actual framing, language, and concerns of the left source. 150-200 words.",
    "source_name": "{left_article.source_name}",
    "source_url": "{left_article.url}"
  }},
  "right": {{
    "headline": "headline written in the framing of the right source",
    "summary": "2-3 sentence summary of how right media covered this",
    "body": "HTML string with 3-4 <p> tags OR a <ul>/<ol> list. Reflect the actual framing, language, and concerns of the right source. 150-200 words.",
    "source_name": "{right_article.source_name}",
    "source_url": "{right_article.url}"
  }}
}}

Rules:
- Never invent facts. Only use what is in the provided articles.
- Do not use &nbsp; in HTML. Use regular spaces only.
- Do not add any text outside the JSON object.
- The left and right bodies must sound genuinely different in tone and focus."""
        
        try:
            if self.provider == "anthropic":
                response_text = self._call_anthropic(system_prompt, user_prompt)
            elif self.provider == "openai":
                response_text = self._call_openai(system_prompt, user_prompt)
            elif self.provider == "ollama":
                response_text = self._call_ollama(system_prompt, user_prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            result = self._parse_response(response_text, left_article, right_article)
            if result:
                logger.info(f"Successfully generated perspectives for topic: {result.topic}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            return self._retry_with_strict_prompt(cluster, left_article_text, right_article_text)
        except Exception as e:
            logger.error(f"Error calling AI API: {e}")
            return None

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "format": "json"
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def _parse_response(self, response_text: str, left_article: RawArticle, right_article: RawArticle) -> Optional[PerspectiveResult]:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            logger.error("No JSON found in response")
            return None
        
        json_str = response_text[json_start:json_end]
        data = json.loads(json_str)
        
        return PerspectiveResult(
            topic=data["topic"],
            left_headline=data["left"]["headline"],
            left_summary=data["left"]["summary"],
            left_body=data["left"]["body"],
            left_source_name=data["left"]["source_name"],
            left_source_url=data["left"]["source_url"],
            right_headline=data["right"]["headline"],
            right_summary=data["right"]["summary"],
            right_body=data["right"]["body"],
            right_source_name=data["right"]["source_name"],
            right_source_url=data["right"]["source_url"]
        )

    def _retry_with_strict_prompt(self, cluster: Cluster, left_article_text: str, right_article_text: str) -> Optional[PerspectiveResult]:
        """Retry with a stricter prompt if JSON parsing failed."""
        left_article = cluster.left_articles[0]
        right_article = cluster.right_articles[0]
        
        strict_prompt = f"""You must return ONLY valid JSON. No other text.

LEFT: {left_article.title}
Content: {left_article_text}

RIGHT: {right_article.title}
Content: {right_article_text}

Return exactly this JSON structure:
{{
  "topic": "neutral headline",
  "left": {{
    "headline": "left framing",
    "summary": "2-3 sentences",
    "body": "HTML with 3-4 <p> tags, 150-200 words",
    "source_name": "{left_article.source_name}",
    "source_url": "{left_article.url}"
  }},
  "right": {{
    "headline": "right framing",
    "summary": "2-3 sentences",
    "body": "HTML with 3-4 <p> tags, 150-200 words",
    "source_name": "{right_article.source_name}",
    "source_url": "{right_article.url}"
  }}
}}"""
        
        try:
            if self.provider == "anthropic":
                response_text = self._call_anthropic("", strict_prompt)
            elif self.provider == "openai":
                response_text = self._call_openai("", strict_prompt)
            elif self.provider == "ollama":
                response_text = self._call_ollama("", strict_prompt)
            
            result = self._parse_response(response_text, left_article, right_article)
            return result
            
        except Exception as e:
            logger.error(f"Retry also failed: {e}")
            return None

    def close(self):
        """Close the client if needed."""
        if self.provider == "ollama" and hasattr(self, 'client'):
            self.client.close()
