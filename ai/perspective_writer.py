import anthropic
from openai import OpenAI
import httpx
import google.generativeai as genai
from dataclasses import dataclass
from typing import Optional
import json
import logging
import re
import time

from clustering import Cluster
from scraper import RawArticle

logger = logging.getLogger(__name__)

MAX_TOKENS = 4000
MAX_RETRIES = 3
RETRY_BACKOFF = 5

ALLOWED_CATEGORIES = {
    "politics", "economy", "world", "sports",
    "entertainment", "technology", "health", "environment", "other"
}


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
    SUPPORTED_PROVIDERS = {"anthropic", "openai", "ollama", "gemini"}

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        api_key: str = None,
        openai_api_key: str = None,
        gemini_api_key: str = None,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.provider = provider
        self.model = model

        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Choose from {self.SUPPORTED_PROVIDERS}")

        if provider == "anthropic":
            if not api_key:
                raise ValueError("Anthropic API key is required")
            self.client = anthropic.Anthropic(api_key=api_key)

        elif provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required")
            self.client = OpenAI(api_key=openai_api_key)

        elif provider == "gemini":
            if not gemini_api_key:
                raise ValueError("Gemini API key is required")
            genai.configure(api_key=gemini_api_key)
            self.client = genai.GenerativeModel(
                model_name=model,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    max_output_tokens=MAX_TOKENS,
                    temperature=0.4,
                ),
            )

        elif provider == "ollama":
            self.ollama_base_url = ollama_base_url
            self.client = httpx.Client(timeout=120.0)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_perspectives(
        self,
        cluster: Cluster,
        left_article_text: str,
        right_article_text: str,
    ) -> Optional[PerspectiveResult]:
        """Generate left and right perspectives using the configured AI provider."""
        if not cluster.left_articles or not cluster.right_articles:
            logger.error("Cluster must have at least one left and one right article")
            return None

        left_article = cluster.left_articles[0]
        right_article = cluster.right_articles[0]

        system_prompt, user_prompt = self._build_prompts(
            left_article, right_article, left_article_text, right_article_text
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response_text = self._dispatch(system_prompt, user_prompt)
                result = self._parse_response(response_text, left_article, right_article)
                if result:
                    logger.info(f"Generated perspectives for: {result.topic!r} (attempt {attempt})")
                    return result
                logger.warning(f"Attempt {attempt}: parse returned None, retrying...")
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt}: JSON decode error — {e}")
            except Exception as e:
                logger.error(f"Attempt {attempt}: API error — {e}")
                if attempt == MAX_RETRIES:
                    return None

            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF ** attempt
                logger.info(f"Waiting {wait}s before retry...")
                time.sleep(wait)
                # Switch to strict prompt on last retry
                if attempt == MAX_RETRIES - 1:
                    user_prompt = self._build_strict_prompt(left_article, right_article)
                    system_prompt = ""

        return None

    # ------------------------------------------------------------------ #
    #  Prompt builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_prompts(
        self,
        left_article: RawArticle,
        right_article: RawArticle,
        left_article_text: str,
        right_article_text: str,
    ) -> tuple[str, str]:
        system_prompt = (
            'You are an expert journalist and editorial writer for "Spectrum News", '
            "a news aggregator that presents every story from two distinct perspectives — "
            "one progressive and one conservative — so readers can understand the full picture. "
            "Transform raw news reports into compelling, human-written articles that genuinely reflect "
            "how different viewpoints would frame the same story.\n\n"
            "Your writing must be:\n"
            "- Engaging and readable (not dry or robotic)\n"
            "- Journalistically professional but with clear perspective framing\n"
            "- Factually accurate to the source material\n"
            "- Distinctly different between the two perspectives\n"
            "- Written in active voice with strong verbs\n"
            "- Grounded in specific quotes, statistics, and details from the sources\n"
            "- Applicable to ANY topic — politics, economy, health, environment, technology, sports, society\n\n"
            "You are NOT a political commentator. You are a neutral editor who can faithfully "
            "represent how a progressive reader and a conservative reader would each experience "
            "the same story — whether it is about a budget, a court ruling, a climate report, "
            "a sports controversy, or a technology policy."
        )

        user_prompt = f"""Transform these raw news reports into compelling editorial pieces.

        PROGRESSIVE SOURCE ({left_article.source_name}):
        Headline: {left_article.title}
        Full Article:
        {left_article_text}

        CONSERVATIVE SOURCE ({right_article.source_name}):
        Headline: {right_article.title}
        Full Article:
        {right_article_text}

        Your task:
        1. Extract key facts, quotes, statistics, and specific details from both articles
        2. Write a compelling neutral headline that captures the essence (8-12 words)
        3. Write an engaging 2-3 sentence description that hooks the reader from a neutral standpoint
        4. Generate 5 relevant tags for categorisation
        5. Assign the most appropriate category from: politics, economy, world, sports, entertainment, technology, health, environment, other
        6. Write TWO distinctly different editorial pieces — one progressive, one conservative

        ─────────────────────────────────────────
        PERSPECTIVE FRAMING RULES
        ─────────────────────────────────────────
        The two perspectives MUST feel genuinely different in:

        Tone:
        Progressive voice — empathetic, community-focused, reform-oriented
        Conservative voice — pragmatic, stability-focused, institution-trusting

        Emphasis:
        Progressive — highlights who is affected and systemic causes
        Conservative — highlights what works, costs, and practical tradeoffs

        Framing:
        Progressive asks "who does this leave behind?"
        Conservative asks "does this actually work and at what cost?"

        These rules apply regardless of topic:
        Health       → progressive: access and equity        | conservative: cost and personal choice
        Environment  → progressive: climate urgency          | conservative: economic impact of policy
        Technology   → progressive: safety and privacy       | conservative: innovation and growth
        Sports       → progressive: inclusion and representation | conservative: tradition and merit
        Economy      → progressive: worker and consumer impact  | conservative: growth and efficiency
        Politics     → progressive: accountability and rights   | conservative: order and governance

        ─────────────────────────────────────────
        CRITICAL — THE SUMMARY IS THE HEART OF THIS APP
        ─────────────────────────────────────────
        The summary is the FIRST thing readers see. It must make them immediately feel
        the difference in perspective — like reading two different realities of the same event.

        Each summary must be a punchy HTML snippet structured like this:

        PROGRESSIVE summary rules:
        - Open with the human cost, the gap, or the systemic concern
        - Use emotionally resonant but journalistic language (e.g. "left behind", "overlooked", "at risk")
        - Name specific affected groups — workers, patients, communities, the vulnerable
        - End with a pointed question or call for accountability
        - Tone: urgent, empathetic, reform-minded but still journalistic

        CONSERVATIVE summary rules:
        - Open with the policy rationale, the national interest, or the practical reality
        - Use confident, results-oriented language (e.g. "decisive", "long overdue", "pragmatic")
        - Emphasise order, efficiency, growth, security, or proven systems
        - End with a forward-looking assertion about stability or progress
        - Tone: authoritative, solutions-focused, skeptical of rushed change

        HTML format for BOTH summaries — use exactly this structure:
        <p class="summary-lead"><strong>[One punchy sentence that is the ideological hook]</strong></p>
        <p>[2-3 sentences expanding with specific facts, names, numbers from the article]</p>
        <p class="summary-question"><em>[Closing line — a pointed question (progressive) or bold assertion (conservative)]</em></p>

        EXAMPLE — factory closure story:

        Progressive summary:
        <p class="summary-lead"><strong>Three thousand workers woke up jobless today while the government celebrated "ease of doing business."</strong></p>
        <p>The abrupt shutdown of Bharat Steel's Nagpur plant has left daily-wage labourers with no severance, no notice, and no safety net — exposing the hollow promises of industrial policy that prioritises investor returns over worker rights.</p>
        <p class="summary-question"><em>Who is held accountable when the system fails the people it claims to protect?</em></p>

        Conservative summary:
        <p class="summary-lead"><strong>Bharat Steel's Nagpur closure is a wake-up call for the labour reforms business has demanded for a decade.</strong></p>
        <p>The plant's shutdown, driven by uncompetitive cost structures and rigid labour laws, underscores why India must modernise its industrial framework to attract the next generation of manufacturing investment.</p>
        <p class="summary-question"><em>Without bold reform, India risks watching these jobs migrate to Vietnam and Bangladesh permanently.</em></p>

        Notice how both use the SAME facts but feel like completely different stories.
        ─────────────────────────────────────────

        Write like a skilled journalist who genuinely holds that perspective. Use specific
        details, quotes, and numbers from the source articles. Do NOT be generic —
        every sentence should feel like it was written by someone who cares deeply about
        their worldview.

        Return ONLY a valid JSON object — no markdown, no extra text, no code fences.
        Ensure the JSON is complete and valid — do not truncate mid-response.
        All apostrophes inside string values must be escaped or replaced with unicode equivalent.
        All HTML inside JSON strings must have quotes escaped as \\".

        {{
        "topic": "neutral headline 8-12 words capturing the core story",
        "description": "2-3 sentence neutral hook that intrigues readers from both sides",
        "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
        "category": "one of: politics, economy, world, sports, entertainment, technology, health, environment, other",
        "left": {{
            "headline": "progressive framing headline — lead with human impact or systemic concern",
            "summary": "<p class=\\"summary-lead\\"><strong>[hook]</strong></p><p>[facts with specifics]</p><p class=\\"summary-question\\"><em>[pointed question]</em></p>",
            "body": "200-250 words in HTML <p> tags written as a progressive journalist would write it",
            "source_name": "{left_article.source_name}",
            "source_url": "{left_article.url}"
        }},
        "right": {{
            "headline": "conservative framing headline — lead with policy rationale or national interest",
            "summary": "<p class=\\"summary-lead\\"><strong>[hook]</strong></p><p>[facts with specifics]</p><p class=\\"summary-question\\"><em>[bold assertion]</em></p>",
            "body": "200-250 words in HTML <p> tags written as a conservative journalist would write it",
            "source_name": "{right_article.source_name}",
            "source_url": "{right_article.url}"
        }}
        }}"""

        return system_prompt, user_prompt

    def _build_strict_prompt(
        self, left_article: RawArticle, right_article: RawArticle
    ) -> str:
        """Minimal prompt used as a last-resort retry."""
        return (
            "Output ONLY a JSON object. No explanation, no markdown, no extra text.\n\n"
            f"Story 1: {left_article.title}\n"
            f"Story 2: {right_article.title}\n\n"
            "Use this exact structure:\n"
            '{"topic":"headline","description":"desc","tags":["t1","t2","t3","t4","t5"],'
            '"category":"politics",'
            f'"left":{{"headline":"h","summary":"s","body":"<p>body</p>",'
            f'"source_name":"{left_article.source_name}","source_url":"{left_article.url}"}},'
            f'"right":{{"headline":"h","summary":"s","body":"<p>body</p>",'
            f'"source_name":"{right_article.source_name}","source_url":"{right_article.url}"}}'
            "}"
        )

    # ------------------------------------------------------------------ #
    #  Provider dispatch                                                   #
    # ------------------------------------------------------------------ #

    def _dispatch(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        elif self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        raise ValueError(f"Unsupported provider: {self.provider}")

    # ------------------------------------------------------------------ #
    #  Provider implementations                                           #
    # ------------------------------------------------------------------ #

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """
        Gemini supports system instructions separately.
        response_mime_type='application/json' is set in the constructor so the
        model always returns valid JSON — no regex stripping needed.
        """
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        response = self.client.generate_content(full_prompt)
        return response.text

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},  # fixed typo: was "temperatue"
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")

    # ------------------------------------------------------------------ #
    #  Response parsing                                                    #
    # ------------------------------------------------------------------ #

    def _parse_response(
        self,
        response_text: str,
        left_article: RawArticle,
        right_article: RawArticle,
    ) -> Optional[PerspectiveResult]:

        logger.info(f"The response text is : {response_text}")

        # Strip whitespace, BOM, and markdown fences
        cleaned = response_text.strip().lstrip('\ufeff')
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned.strip())

        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1

        if json_start == -1 or json_end == 0:
            logger.error(f"No JSON object found in response. Preview: {response_text[:200]!r}")
            return None

        json_str = cleaned[json_start:json_end]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}. Snippet: {json_str[:300]!r}")
            raise  # bubble up so retry logic triggers

        category = self._normalise_category(data.get("category", "other"))

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
            right_source_url=data.get("right", {}).get("source_url", right_article.url),
        )

    @staticmethod
    def _normalise_category(raw: str) -> str:
        candidate = re.split(r"[/\s]", raw.lower().strip())[0]
        return candidate if candidate in ALLOWED_CATEGORIES else "other"

    # ------------------------------------------------------------------ #
    #  Cleanup                                                             #
    # ------------------------------------------------------------------ #

    def close(self):
        if self.provider == "ollama" and hasattr(self, "client"):
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()