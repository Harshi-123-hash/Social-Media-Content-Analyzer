from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    engagement_score: int
    word_count: int
    hashtag_count: int
    call_to_action: bool
    question: bool
    sentiment: str
    caption_length: str
    suggestions: List[str]
    improved_caption: str
    hashtags: List[str]


# ---------------------------------------------------------------------------
# Basic NLP dictionaries
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "benefit",
    "brilliant",
    "clean",
    "easy",
    "excellent",
    "excited",
    "fantastic",
    "good",
    "great",
    "happy",
    "innovative",
    "love",
    "new",
    "powerful",
    "success",
    "successful",
    "useful",
    "wonderful",
}

NEGATIVE_WORDS = {
    "bad",
    "boring",
    "difficult",
    "disappointing",
    "hate",
    "issue",
    "problem",
    "poor",
    "sad",
    "slow",
    "terrible",
    "worst",
}


CTA_PATTERNS = (
    r"\btry\b",
    r"\bshop\b",
    r"\bbuy\b",
    r"\bdiscover\b",
    r"\blearn more\b",
    r"\bget started\b",
    r"\bsign up\b",
    r"\bdownload\b",
    r"\bvisit\b",
    r"\bjoin\b",
    r"\bcontact\b",
    r"\border\b",
    r"\bbook\b",
    r"\bsubscribe\b",
    r"\btell us\b",
    r"\bshare\b",
    r"\bcomment\b",
)


# Words that are too generic to become useful hashtags.
STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "into",
    "for",
    "your",
    "our",
    "you",
    "we",
    "they",
    "their",
    "its",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "as",
    "it",
    "i",
    "me",
    "my",
    "us",
    "will",
    "can",
    "could",
    "would",
    "should",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "not",
    "very",
    "more",
    "most",
    "also",
    "just",
    "about",
    "than",
    "so",
    "today",
    "here",
    "there",
    "what",
    "when",
    "where",
    "who",
    "how",
}


# ---------------------------------------------------------------------------
# Topic dictionary
#
# These are NOT the final hashtags.
# They are used to understand the context of the uploaded caption.
# ---------------------------------------------------------------------------

TOPIC_HASHTAGS: Dict[str, List[str]] = {
    "eco": [
        "eco",
        "eco-friendly",
        "ecofriendly",
        "sustainable",
        "sustainability",
        "environment",
        "environmental",
        "green",
        "recycled",
        "recyclable",
        "plastic-free",
        "plastic",
        "waste",
        "renewable",
        "earth",
        "planet",
    ],

    "water_bottle": [
        "water bottle",
        "bottle",
        "hydration",
        "reusable bottle",
        "reusable",
        "drinking water",
        "water",
    ],

    "technology": [
        "technology",
        "tech",
        "software",
        "app",
        "application",
        "digital",
        "computer",
        "programming",
        "developer",
        "development",
        "artificial intelligence",
        "ai",
        "machine learning",
        "automation",
        "cloud",
        "data",
    ],

    "business": [
        "business",
        "startup",
        "company",
        "entrepreneur",
        "entrepreneurship",
        "marketing",
        "sales",
        "customer",
        "customers",
        "brand",
        "product",
        "launch",
    ],

    "fitness": [
        "fitness",
        "workout",
        "exercise",
        "gym",
        "training",
        "running",
        "running",
        "strength",
        "muscle",
        "health",
        "wellness",
        "yoga",
    ],

    "food": [
        "food",
        "recipe",
        "cooking",
        "cook",
        "meal",
        "breakfast",
        "lunch",
        "dinner",
        "restaurant",
        "delicious",
        "vegan",
        "vegetarian",
        "dessert",
    ],

    "travel": [
        "travel",
        "trip",
        "vacation",
        "holiday",
        "tour",
        "tourism",
        "destination",
        "hotel",
        "flight",
        "beach",
        "mountain",
        "adventure",
    ],

    "education": [
        "education",
        "student",
        "students",
        "school",
        "college",
        "university",
        "learning",
        "learn",
        "course",
        "teacher",
        "study",
        "exam",
        "tutorial",
    ],

    "fashion": [
        "fashion",
        "clothing",
        "clothes",
        "dress",
        "style",
        "outfit",
        "shoes",
        "beauty",
        "skincare",
        "makeup",
        "collection",
    ],

    "real_estate": [
        "real estate",
        "property",
        "home",
        "house",
        "apartment",
        "realty",
        "mortgage",
        "rent",
        "buying",
        "selling",
    ],
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _extract_hashtags(text: str) -> List[str]:
    return re.findall(r"(?<!\w)#[A-Za-z0-9_]+", text)


def _has_call_to_action(text: str) -> bool:
    lowered = text.lower()

    return any(
        re.search(pattern, lowered)
        for pattern in CTA_PATTERNS
    )


def _detect_sentiment(text: str) -> str:
    words = set(
        re.findall(
            r"\b[a-zA-Z]+\b",
            text.lower(),
        )
    )

    positive_score = len(words & POSITIVE_WORDS)
    negative_score = len(words & NEGATIVE_WORDS)

    if positive_score > negative_score:
        return "Positive"

    if negative_score > positive_score:
        return "Negative"

    return "Neutral"


def _caption_length(word_count: int) -> str:
    if word_count < 8:
        return "Too short"

    if word_count <= 30:
        return "Good"

    if word_count <= 80:
        return "Long"

    return "Too long"


# ---------------------------------------------------------------------------
# Dynamic hashtag generation
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """
    Convert text into a searchable lowercase representation.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_topic_match(text: str, keyword: str) -> bool:
    """
    Check whether a topic keyword appears in the caption.

    Multi-word phrases are checked directly.
    Single words use word boundaries to avoid accidental matches.
    """
    if " " in keyword:
        return keyword in text

    return bool(
        re.search(
            rf"\b{re.escape(keyword)}\b",
            text,
        )
    )


def _topic_matches(text: str) -> List[str]:
    """
    Find the topics represented in the caption.
    """
    normalized = _normalise_text(text)
    matched_topics: List[str] = []

    for topic, keywords in TOPIC_HASHTAGS.items():
        if any(
            _has_topic_match(normalized, keyword)
            for keyword in keywords
        ):
            matched_topics.append(topic)

    return matched_topics


def _keyword_hashtags(text: str, limit: int = 3) -> List[str]:
    """
    Extract useful repeated/content-heavy words from the caption.

    This provides a fallback when the caption does not match one
    of the predefined topic groups.
    """
    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]{3,}\b",
        text.lower(),
    )

    frequency: Dict[str, int] = {}

    for word in words:
        clean_word = word.replace("-", "")

        if clean_word in STOP_WORDS:
            continue

        if len(clean_word) < 4:
            continue

        frequency[word] = frequency.get(word, 0) + 1

    sorted_words = sorted(
        frequency.items(),
        key=lambda item: (-item[1], item[0]),
    )

    hashtags: List[str] = []

    for word, _count in sorted_words:
        hashtag_word = re.sub(
            r"[^a-zA-Z0-9]",
            "",
            word,
        )

        if not hashtag_word:
            continue

        hashtag = f"#{hashtag_word.capitalize()}"

        if hashtag.lower() not in {
            item.lower()
            for item in hashtags
        }:
            hashtags.append(hashtag)

        if len(hashtags) >= limit:
            break

    return hashtags


def _generate_hashtags(
    text: str,
    existing_hashtags: List[str],
) -> List[str]:
    """
    Generate context-aware hashtags based on the actual caption.

    Existing hashtags are preserved first. New hashtags are generated
    from matching topics and keyword extraction.
    """
    result: List[str] = []

    # Keep hashtags that the user actually provided.
    for hashtag in existing_hashtags:
        if hashtag.lower() not in {
            item.lower()
            for item in result
        }:
            result.append(hashtag)

    topics = _topic_matches(text)

    topic_hashtags: Dict[str, List[str]] = {
        "eco": [
            "#EcoFriendly",
            "#Sustainability",
            "#GreenLiving",
            "#Environment",
            "#PlasticFree",
        ],
        "water_bottle": [
            "#Hydration",
            "#ReusableBottle",
            "#WaterBottle",
            "#HealthyLiving",
        ],
        "technology": [
            "#Technology",
            "#Tech",
            "#Innovation",
            "#Digital",
            "#Software",
        ],
        "business": [
            "#Business",
            "#Entrepreneurship",
            "#Marketing",
            "#Startup",
            "#BusinessGrowth",
        ],
        "fitness": [
            "#Fitness",
            "#Workout",
            "#HealthyLifestyle",
            "#Exercise",
            "#Wellness",
        ],
        "food": [
            "#Food",
            "#Foodie",
            "#Cooking",
            "#Recipe",
            "#HealthyFood",
        ],
        "travel": [
            "#Travel",
            "#TravelGoals",
            "#Adventure",
            "#Vacation",
            "#Explore",
        ],
        "education": [
            "#Education",
            "#Learning",
            "#Students",
            "#Study",
            "#Knowledge",
        ],
        "fashion": [
            "#Fashion",
            "#Style",
            "#Outfit",
            "#FashionStyle",
            "#Beauty",
        ],
        "real_estate": [
            "#RealEstate",
            "#Property",
            "#Home",
            "#RealEstateInvestment",
            "#DreamHome",
        ],
    }

    # Add topic-specific hashtags.
    for topic in topics:
        for hashtag in topic_hashtags.get(topic, []):
            if len(result) >= 5:
                break

            if hashtag.lower() not in {
                item.lower()
                for item in result
            }:
                result.append(hashtag)

        if len(result) >= 5:
            break

    # If the topic dictionary did not provide enough hashtags,
    # use actual keywords from the uploaded caption.
    if len(result) < 3:
        fallback_hashtags = _keyword_hashtags(
            text,
            limit=5 - len(result),
        )

        for hashtag in fallback_hashtags:
            if hashtag.lower() not in {
                item.lower()
                for item in result
            }:
                result.append(hashtag)

            if len(result) >= 5:
                break

    return result[:5]


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

def _build_suggestions(
    word_count: int,
    hashtag_count: int,
    has_cta: bool,
    has_question: bool,
    sentiment: str,
) -> List[str]:

    suggestions: List[str] = []

    if word_count < 8:
        suggestions.append(
            "Add a stronger opening sentence with useful context."
        )

    elif word_count > 80:
        suggestions.append(
            "Use shorter sentences and remove unnecessary words for readability."
        )

    if hashtag_count == 0:
        suggestions.append(
            "Add 3–5 relevant hashtags based on the actual post topic."
        )

    elif hashtag_count < 3:
        suggestions.append(
            "Consider adding more relevant hashtags to improve discoverability."
        )

    elif hashtag_count > 5:
        suggestions.append(
            "Reduce the number of hashtags and keep only the most relevant ones."
        )

    if not has_cta:
        suggestions.append(
            "Add a clear call-to-action such as 'Try it today', 'Learn more', or 'Tell us what you think'."
        )

    if not has_question:
        suggestions.append(
            "Ask a question to encourage comments and discussion."
        )

    if sentiment == "Negative":
        suggestions.append(
            "Use a more positive and benefit-focused tone."
        )

    if not suggestions:
        suggestions.append(
            "Your caption is already strong. Test different hooks and calls-to-action to improve engagement."
        )

    return suggestions


# ---------------------------------------------------------------------------
# Improved caption
# ---------------------------------------------------------------------------

def _remove_existing_hashtags(text: str) -> str:
    """
    Remove hashtags from the body before rebuilding the improved caption.
    """
    return re.sub(
        r"(?<!\w)#[A-Za-z0-9_]+",
        "",
        text,
    ).strip()


def _build_improved_caption(
    text: str,
    generated_hashtags: List[str],
    has_cta: bool,
    has_question: bool,
) -> str:

    cleaned = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # Remove hashtags from the original text so they aren't duplicated.
    cleaned = _remove_existing_hashtags(cleaned)

    # Remove accidental labels that may have come from PDF extraction.
    cleaned = re.split(
        r"\b(improved caption|suggestions?|analysis|content analysis)\s*:",
        cleaned,
        flags=re.IGNORECASE,
    )[0].strip()

    if not cleaned:
        cleaned = (
            "Excited to share something new with you!"
        )

    # Prevent extremely long generated captions.
    words = cleaned.split()

    if len(words) > 55:
        cleaned = (
            " ".join(words[:55])
            .rstrip(".,!?")
            + "."
        )

    additions: List[str] = []

    if not has_question:
        additions.append(
            "What do you think?"
        )

    if not has_cta:
        additions.append(
            "Try it today and tell us what you think!"
        )

    # IMPORTANT:
    # The original caption is included only ONCE.
    improved_parts = [cleaned]

    if additions:
        improved_parts.extend(additions)

    if generated_hashtags:
        improved_parts.append(
            " ".join(generated_hashtags)
        )

    return " ".join(improved_parts)


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

def analyze_content(text: str) -> AnalysisResult:
    """
    Analyze social media content and generate engagement improvements.
    """

    text = re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()

    word_count = _count_words(text)

    existing_hashtags = _extract_hashtags(text)

    # Generate hashtags from the actual text.
    generated_hashtags = _generate_hashtags(
        text=text,
        existing_hashtags=existing_hashtags,
    )

    # Use generated hashtags for the displayed hashtag count.
    hashtag_count = len(generated_hashtags)

    has_cta = _has_call_to_action(text)

    has_question = "?" in text

    sentiment = _detect_sentiment(text)

    caption_length = _caption_length(word_count)

    # Dynamic engagement score.
    score = 40

    if 8 <= word_count <= 80:
        score += 10

    if 15 <= word_count <= 60:
        score += 5

    if 3 <= hashtag_count <= 5:
        score += 15

    elif 1 <= hashtag_count <= 2:
        score += 8

    if has_cta:
        score += 15

    if has_question:
        score += 10

    if sentiment == "Positive":
        score += 5

    elif sentiment == "Neutral":
        score += 3

    score = max(
        0,
        min(score, 100),
    )

    suggestions = _build_suggestions(
        word_count=word_count,
        hashtag_count=hashtag_count,
        has_cta=has_cta,
        has_question=has_question,
        sentiment=sentiment,
    )

    improved_caption = _build_improved_caption(
        text=text,
        generated_hashtags=generated_hashtags,
        has_cta=has_cta,
        has_question=has_question,
    )

    return AnalysisResult(
        engagement_score=score,
        word_count=word_count,
        hashtag_count=hashtag_count,
        call_to_action=has_cta,
        question=has_question,
        sentiment=sentiment,
        caption_length=caption_length,
        suggestions=suggestions,
        improved_caption=improved_caption,
        hashtags=generated_hashtags,
    )