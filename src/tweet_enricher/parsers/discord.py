"""Discord export file parser and message processing."""

import re
from pathlib import Path
from typing import Dict, List

from tweet_enricher.config import EXCLUDED_TICKERS


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
            Cleaned text
        """
        # Remove noise patterns
        clean = re.sub(r"\{Embed\}", "", text)
        clean = re.sub(r"\{Attachments\}", "", clean)
        clean = re.sub(r"https?://[^\s]+", "", clean)
        clean = re.sub(r"TweetShift[^\n]*", "", clean)
        clean = re.sub(r"Powered by [^\n]+", "", clean)
        clean = re.sub(r"📷\d+", "", clean)
        clean = re.sub(r"\[Tweeted\][^\n]*", "", clean)

        # Clean up whitespace
        clean = "\n".join(line.strip() for line in clean.split("\n") if line.strip())
        clean = clean.strip()

        # Remove escape characters (Discord escapes special chars)
        clean = re.sub(r"\\([()[\]{}.,!?\-\'#+_])", r"\1", clean)

        # Replace newlines with spaces for CSV
        clean = " ".join(clean.split("\n"))

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
            if ("\u0041" <= char <= "\u007A") or ("\u00C0" <= char <= "\u024F") or char in "\u00aa\u00ba":
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


class DiscordParser:
    """Parses Discord channel export files."""

    TIMESTAMP_PATTERN = re.compile(r"\[(\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2})\]\s+(.*)")

    def __init__(self, processor: MessageProcessor, categorizer: MessageCategorizer):
        """
        Initialize Discord parser.

        Args:
            processor: Message processor instance
            categorizer: Message categorizer instance
        """
        self.processor = processor
        self.categorizer = categorizer

    def parse_file(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Parse Discord export file.

        Args:
            file_path: Path to Discord export file

        Returns:
            List of parsed messages
        """
        messages = []
        current_message = None

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip header
        start_idx = self._find_start_index(lines)

        for i in range(start_idx, len(lines)):
            line = lines[i]
            timestamp_match = self.TIMESTAMP_PATTERN.match(line)

            if timestamp_match:
                # Save previous message
                if current_message:
                    messages.extend(self._process_message(current_message))

                # Start new message
                current_message = {
                    "timestamp": timestamp_match.group(1),
                    "username": timestamp_match.group(2).strip(),
                    "text_lines": [],
                }
            elif current_message:
                current_message["text_lines"].append(line)

        # Process last message
        if current_message:
            messages.extend(self._process_message(current_message))

        return messages

    def _find_start_index(self, lines: List[str]) -> int:
        """Find the index where messages start."""
        for i, line in enumerate(lines):
            if line.strip().startswith("["):
                return i
        return 0

    def _clean_author(self, username: str) -> str:
        """Remove service suffix from author name (e.g., '• TweetShift')."""
        return username.split(" • ")[0].strip()

    def _extract_content_from_embed(self, embed_text: str) -> str:
        """Extract tweet content from inside an {Embed} block."""
        lines = embed_text.strip().split("\n")
        content_lines = []
        skip_next_url = True  # Skip first URL line (tweet URL)

        for line in lines:
            line = line.strip()
            # Skip author line: "Author (@handle) ✧" or "Author (@handle)"
            if re.match(r"^[\w\s]+\(@\w+\)(\s*✧)?$", line):
                continue
            # Skip first URL line only (tweet URL), preserve subsequent URLs in content
            if line.startswith("https://") or line.startswith("http://"):
                if skip_next_url:
                    skip_next_url = False
                    continue
                # Keep subsequent URLs - they may be part of the tweet content
            # Skip footer patterns
            if line in ["TweetShift", "X"] or line.startswith("TweetShift •"):
                continue
            # Skip empty lines at start
            if not content_lines and not line:
                continue
            content_lines.append(line)

        return "\n".join(content_lines)

    def _process_message(self, message: Dict) -> List[Dict[str, str]]:
        """Process a single message and extract data."""
        full_text = "".join(message["text_lines"]).strip()

        # Split at {Embed} to handle duplicates correctly
        if "{Embed}" in full_text:
            parts = full_text.split("{Embed}", 1)
            before_embed = parts[0].strip()
            inside_embed = parts[1].strip() if len(parts) > 1 else ""

            # Remove [Tweeted](...) link from before_embed
            before_embed = re.sub(r"\[Tweeted\]\([^)]+\)\s*", "", before_embed).strip()

            # If substantial content before {Embed}, use it (Pattern B - long tweets)
            # Otherwise, extract from inside {Embed} (Pattern A - short tweets)
            if len(before_embed) > 50:
                content_text = before_embed
            else:
                content_text = self._extract_content_from_embed(inside_embed)
        else:
            content_text = full_text

        # Extract components from full_text (for URL extraction)
        tickers = self.processor.extract_tickers(full_text)
        tweet_url = self.processor.extract_tweet_url(full_text)
        clean_text = self.processor.clean_text(content_text)

        # Validate
        if not self.processor.is_valid(clean_text):
            return []

        # Categorize
        category = self.categorizer.categorize(clean_text)

        # Create result entries
        result = []
        if tickers:
            for ticker in tickers:
                # Skip excluded tickers (problematic symbols that consistently fail in IBKR)
                if ticker and ticker not in EXCLUDED_TICKERS:
                    result.append(
                        {
                            "timestamp": message["timestamp"],
                            "author": self._clean_author(message["username"]),
                            "ticker": ticker,
                            "tweet_url": tweet_url,
                            "category": category,
                            "text": clean_text,
                        }
                    )

        return result


class DiscordToCSVConverter:
    """Main converter orchestrating the conversion process."""

    def __init__(self, min_text_length: int = 60, deduplicate: bool = True):
        """
        Initialize converter.

        Args:
            min_text_length: Minimum text length to keep
            deduplicate: Remove duplicate messages
        """
        self.processor = MessageProcessor(min_text_length)
        self.categorizer = MessageCategorizer()
        self.parser = DiscordParser(self.processor, self.categorizer)
        self.deduplicate = deduplicate

    def convert(
        self,
        input_file: Path,
        output_file: Path,
        ticker_filter_file: Path = None,
        verbose: bool = True,
    ) -> Dict[str, int]:
        """
        Convert Discord export to CSV.

        Args:
            input_file: Path to Discord export file
            output_file: Path to output CSV file
            ticker_filter_file: Optional ticker filter CSV
            verbose: Print progress information

        Returns:
            Conversion statistics
        """
        from tweet_enricher.io.csv_writer import CSVWriter, TickerFilter

        if verbose:
            print(f"Reading Discord data from: {input_file}")

        # Parse messages
        messages = self.parser.parse_file(input_file)

        if verbose:
            print(f"Found {len(messages)} messages")

        # Load ticker filter if provided
        ticker_filter = None
        if ticker_filter_file and ticker_filter_file.exists():
            ticker_filter = TickerFilter.load_from_csv(ticker_filter_file)
            if verbose:
                print(f"Loaded {len(ticker_filter)} tickers from filter list")

        # Write to CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Writing to CSV: {output_file}")

        writer = CSVWriter(self.deduplicate, ticker_filter)
        stats = writer.write(messages, output_file)

        if verbose:
            self._print_stats(stats, messages)

        return stats

    def _print_stats(self, stats: Dict[str, int], messages: List[Dict[str, str]]) -> None:
        """Print conversion statistics."""
        print("\nConversion complete!")

        if stats["filtered"] > 0:
            print(f"Filtered out {stats['filtered']} messages (tickers not in filter list)")

        if stats["duplicates"] > 0:
            print(f"Removed {stats['duplicates']} duplicate messages")

        print(f"Written {stats['written']} unique messages to CSV")
