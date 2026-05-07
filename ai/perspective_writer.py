import anthropic
from openai import OpenAI
import httpx
from dataclasses import dataclass
from typing import Optional
import json
import logging
import re

from clustering import Cluster
from scraper import RawArticle

logger = logging.getLogger(__name__)


@dataclass
class PerspectiveResult:
    topic: str
    description: str
    tags: list[str]
    category: str
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
        
        system_prompt = """You are an expert Indian political journalist and editorial writer for "Biased India", a news aggregator that presents the same story from multiple political perspectives. Your job is to transform raw news reports into compelling, human-written style articles that genuinely reflect how different political leanings would frame the same story.

Your writing must be:
- Engaging and readable (not dry or robotic)
- Journalistically professional but with clear ideological framing
- Factually accurate to the source material
- Distinctly different between left and right perspectives
- Written in active voice with strong verbs
- Include specific quotes, statistics, and details from sources"""
        
        user_prompt = f"""You are transforming raw news reports into compelling editorial pieces. Below are two articles from Indian news outlets with different political leanings covering the same story.

LEFT-LEANING SOURCE ({left_article.source_name}):
Headline: {left_article.title}
Full Article: {left_article_text}

RIGHT-LEANING SOURCE ({right_article.source_name}):
Headline: {right_article.title}
Full Article: {right_article_text}

Your task:
1. Extract key facts, quotes, statistics, and specific details from both articles
2. Write a compelling neutral headline that captures the essence
3. Write an engaging 2-3 sentence description that hooks the reader
4. Generate 5 relevant tags for categorization
5. Assign the most appropriate category
6. Write TWO distinctly different editorial pieces - one from a left/progressive framing, one from a right/conservative framing

CRITICAL: The left and right perspectives MUST be genuinely different in:
- Tone and language (left: more social justice focus, right: more economic/nationalist focus)
- What they emphasize (left: impact on people, right: policy/implementation focus)
- How they frame the issues (left: systemic concerns, right: individual responsibility)
- The angle they take on the same facts

Write in a human, journalistic style - not robotic or generic. Use specific details, quotes, and numbers from the articles. Make it read like something written by a skilled journalist who genuinely holds those political views.

Return ONLY a JSON object. No other text before or after. No markdown formatting.

JSON format:
{{
  "topic": "compelling neutral headline (8-12 words, hook the reader)",
  "description": "engaging 2-3 sentence hook that makes people want to read more",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "category": "politics",
  "left": {{
    "headline": "headline reflecting left/progressive framing",
    "summary": "2-3 sentence summary from left perspective",
    "body": "200-250 words in HTML with <p> tags. Write as a left-leaning journalist would. Emphasize social justice, government accountability, impact on ordinary people.",
    "source_name": "{left_article.source_name}",
    "source_url": "{left_article.url}"
  }},
  "right": {{
    "headline": "headline reflecting right/conservative framing",
    "summary": "2-3 sentence summary from right perspective", 
    "body": "200-250 words in HTML with <p> tags. Write as a right-leaning journalist would. Emphasize economic impact, national security, traditional values, policy efficiency.",
    "source_name": "{right_article.source_name}",
    "source_url": "{right_article.url}"
  }}
}}

IMPORTANT JSON RULES:
- Use ONLY the exact field names shown above
- All string values must be in double quotes
- Arrays use square brackets with comma-separated values
- No trailing commas before closing braces
- Category must be ONE word from: politics, economy, world, sports, entertainment, technology, health, environment, other
- No markdown code blocks, no extra text
- Valid JSON only"""
        
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
                "format": "json",
                "options": {"temperatue": 0}
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
        
        # Try to fix common JSON issues before parsing
        # Remove markdown code block markers
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}. Raw text: {json_str[:200]}...")
            return None
        
        # Safely extract category - ensure single word
        category = data.get("category", "politics").lower().strip()
        # Take only first word if multiple words or slashes
        category = re.split(r'[/\s]', category)[0]
        # Validate it's in allowed list
        allowed = ["politics", "economy", "world", "sports", "entertainment", "technology", "health", "environment", "other"]
        if category not in allowed:
            category = "other"
        
        return PerspectiveResult(
            topic=data.get("topic", "News Update"),
            description=data.get("description", ""),
            tags=data.get("tags", ["news", "india", "politics"]),
            category=category,
            left_headline=data.get("left", {}).get("headline", left_article.title),
            left_summary=data.get("left", {}).get("summary", ""),
            left_body=data.get("left", {}).get("body", ""),
            left_source_name=data.get("left", {}).get("source_name", left_article.source_name),
            left_source_url=data.get("left", {}).get("source_url", left_article.url),
            right_headline=data.get("right", {}).get("headline", right_article.title),
            right_summary=data.get("right", {}).get("summary", ""),
            right_body=data.get("right", {}).get("body", ""),
            right_source_name=data.get("right", {}).get("source_name", right_article.source_name),
            right_source_url=data.get("right", {}).get("source_url", right_article.url)
        )

    def _retry_with_strict_prompt(self, cluster: Cluster, left_article_text: str, right_article_text: str) -> Optional[PerspectiveResult]:
        """Retry with a stricter prompt if JSON parsing failed."""
        left_article = cluster.left_articles[0]
        right_article = cluster.right_articles[0]
        
        strict_prompt = f"""Output ONLY a JSON object. No explanation. No markdown. No extra text.

Story 1: {left_article.title}
Story 2: {right_article.title}

Write this exact JSON structure with real content:
{{"topic":"headline here","description":"description here","tags":["tag1","tag2","tag3","tag4","tag5"],"category":"politics","left":{{"headline":"left headline","summary":"left summary","body":"<p>paragraph 1</p><p>paragraph 2</p>","source_name":"{left_article.source_name}","source_url":"{left_article.url}"}},"right":{{"headline":"right headline","summary":"right summary","body":"<p>paragraph 1</p><p>paragraph 2</p>","source_name":"{right_article.source_name}","source_url":"{right_article.url}"}}}}"""
        
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
