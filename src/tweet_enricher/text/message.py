"""Message processing and categorization utilities."""

import re
from typing import List

from tweet_enricher.text.cleaner import clean_for_finbert


class MessageCategorizer:
    """Categorizes messages into predefined categories based on keywords."""

    CATEGORIES = {
        "Earnings": [
            "earnings",
            "quarter",
            "q1",
            "q2",
            "q3",
            "q4",
            "1q",
            "2q",
            "3q",
            "4q",
            "eps",
            "fiscal",
            "fy25",
            "fy26",
            "quarterly",
            "annual report",
            "beats estimate",
            "misses estimate",
            "earnings report",
            "earnings preview",
            "earnings call",
            "net income",
            "gross margin",
            "operating income",
            "ebitda",
            "beats expectations",
            "misses expectations",
        ],
        "Mergers & Acquisitions": [
            "merger",
            "acquisition",
            "acquire",
            "buyout",
            "takeover",
            "bought",
            "purchase",
            "acquiring",
            "merge",
            "consolidation",
            "joint venture",
            "spinoff",
            "spin-off",
            "demerger",
            "divestiture",
            "sale of",
            "sells stake",
        ],
        "Guidance & Forecasts": [
            "guidance",
            "forecast",
            "outlook",
            "projection",
            "expects",
            "anticipates",
            "target",
            "price target",
            "pt ",
            "pt:",
            "estimate",
            "analyst",
            "upgrade",
            "downgrade",
            "rating",
            "initiates coverage",
            "raises",
            "lowers",
            "reaffirms",
            "maintains",
            "sees",
            "guides",
            "2025e",
            "2026e",
            "2027e",
            "forward",
            "next quarter",
        ],
        "Regulatory & Legal": [
            "lawsuit",
            "litigation",
            "legal",
            "court",
            "judge",
            "ruling",
            "regulation",
            "regulatory",
            "sec ",
            "ftc",
            "doj",
            "fda",
            "fcc",
            "antitrust",
            "investigation",
            "probe",
            "fine",
            "penalty",
            "settlement",
            "compliance",
            "violation",
            "sanction",
            "policy",
            "law",
            "ban",
            "approved by",
            "approval",
            "denied",
            "patent",
            "copyright",
        ],
        "Product Launch": [
            "launches",
            "unveiled",
            "announces new",
            "new product",
            "new feature",
            "released",
            "rolls out",
            "introducing",
            "debut",
            "new service",
            "new model",
            "version",
            "update",
            "upgrade to",
            "expands distribution",
            "now available",
            "coming soon",
            "pre-order",
            "delayed",
            "postponed",
            "canceled",
            "cancelled",
            "scrapped",
            "shelved",
            "foldable",
            "launch plans",
        ],
        "Partnerships & Deals": [
            "partnership",
            "partners with",
            "collaboration",
            "deal with",
            "agreement with",
            "contract",
            "signs",
            "expands at",
            "distribution",
            "teams up",
            "joins forces",
            "alliance",
            "works with",
            "in talks",
            "negotiating",
            "discussing",
            "considering",
            "exploring",
            "cloud deal",
            "strategic deal",
            "interest from",
            "interested parties",
        ],
        "Market Data": [
            "market cap",
            "stock price",
            "share price",
            "trading at",
            "trillion",
            "valuation",
            "shares",
            "stock is moving",
            "from $",
            "to $",
            "historical",
            "all-time",
            "since",
            "million to",
            "billion to",
        ],
        "Company Strategy": [
            "plans to",
            "strategy",
            "expansion",
            "expanding",
            "automation",
            "automate",
            "operations",
            "restructuring",
            "reorganization",
            "scaling",
            "growth plan",
            "investing in",
            "focuses on",
            "pivoting",
            "shift",
            "transformation",
            "initiative",
            "robots",
            "replace",
            "workforce",
            "ai-powered",
            "digital transformation",
            "up for sale",
            "received interest",
            "snag the",
            "potentially",
            "reinvent",
            "effort to",
            "development",
            "breakthrough",
        ],
        "Company Metrics": [
            "million users",
            "billion users",
            "customers",
            "subscribers",
            "hosted on",
            "operates",
            "manages",
            "revenue of",
            "sales of",
            "grew by",
            "increased by",
            "decreased by",
            "user growth",
            "active users",
            "monthly active",
            "daily active",
        ],
        "Personnel Changes": [
            "ceo",
            "chief executive",
            "chief financial",
            "cfo",
            "cto",
            "coo",
            "executive",
            "appoints",
            "hired",
            "hires",
            "joins as",
            "steps down",
            "resigns",
            "departing",
            "layoffs",
            "cuts",
            "fires",
            "replaces",
            "names",
            "promoted to",
        ],
        "Breaking News": [
            "breaking",
            "alert",
            "just in",
            "developing",
            "urgent",
            "shutdown",
            "crisis",
            "crash",
            "surge",
            "plunge",
            "spike",
            "record high",
            "record low",
            "halt",
            "suspended",
            "emergency",
            "unprecedented",
        ],
    }

    def categorize(self, text: str) -> str:
        """
        Categorize message based on keyword matching.

        Args:
            text: Message text to categorize

        Returns:
            Category name
        """
        text_lower = text.lower()

        for category, keywords in self.CATEGORIES.items():
            if any(keyword in text_lower for keyword in keywords):
                return category

        return "Other"


class MessageProcessor:
    """Processes and cleans message text."""

    def __init__(self, min_text_length: int = 60):
        """
        Initialize message processor.

        Args:
            min_text_length: Minimum text length to keep
        """
        self.min_text_length = min_text_length

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize message text.

        Args:
            text: Raw message text

        Returns:
            Cleaned text optimized for FinBERT processing
        """
        # Remove URLs
        clean = re.sub(r"https?://[^\s]+", "", text)

        # Clean up whitespace
        clean = "\n".join(line.strip() for line in clean.split("\n") if line.strip())
        clean = clean.strip()

        # Remove escape characters
        clean = re.sub(r"\\([()[\]{}.,!?\-\'#+_])", r"\1", clean)

        # Replace newlines with spaces
        clean = " ".join(clean.split("\n"))

        # Apply FinBERT-optimized cleaning (Unicode normalization, emoji mapping, etc.)
        clean = clean_for_finbert(clean)

        return clean

    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract ticker symbols from text.

        Args:
            text: Message text

        Returns:
            List of ticker symbols
        """
        return re.findall(r"\$([A-Z][A-Z0-9]*)", text)

    def extract_tweet_url(self, text: str) -> str:
        """
        Extract tweet URL from message.

        Args:
            text: Message text

        Returns:
            Tweet URL or empty string
        """
        match = re.search(r"(https?://(?:twitter\.com|x\.com)/[^\s)]+)", text)
        return match.group(1) if match else ""

    def is_english(self, text: str) -> bool:
        """
        Check if text is primarily English (no Hebrew/Arabic/other non-Latin scripts).

        Args:
            text: Text to check

        Returns:
            True if text appears to be English
        """
        if not text:
            return False

        # Count characters by type
        latin_count = 0
        non_latin_count = 0

        for char in text:
            # Skip whitespace, digits, punctuation
            if char.isspace() or char.isdigit() or not char.isalpha():
                continue

            # Check if character is in Latin script (Basic Latin + Latin Extended)
            # Include U+00AA (ª) and U+00BA (º) from Latin-1 Supplement
            if ("\u0041" <= char <= "\u007a") or ("\u00c0" <= char <= "\u024f") or char in "\u00aa\u00ba":
                latin_count += 1
            else:
                non_latin_count += 1

        # If more than 20% non-Latin characters, consider it non-English
        total_alpha = latin_count + non_latin_count
        if total_alpha == 0:
            return True  # No alphabetic characters (e.g., all numbers/symbols)

        return (non_latin_count / total_alpha) < 0.2

    def is_valid(self, text: str) -> bool:
        """
        Check if text meets minimum quality standards.

        Args:
            text: Cleaned text

        Returns:
            True if text is valid
        """
        return bool(text and len(text) >= self.min_text_length and self.is_english(text))

