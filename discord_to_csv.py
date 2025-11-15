#!/usr/bin/env python3
"""
Discord to CSV Converter - Object-Oriented CLI Tool

Converts Discord channel exports to structured CSV with ticker extraction,
categorization, and data quality features.

Usage:
    python discord_to_csv.py --help
    python discord_to_csv.py --input discord_data.txt --output messages.csv
    python discord_to_csv.py --input discord_data.txt --filter tickers.csv --no-dedup
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional
from collections import Counter
import pytz


class MessageCategorizer:
    """Categorizes messages into predefined categories based on keywords."""

    CATEGORIES = {
        'Earnings': [
            'earnings', 'quarter', 'q1', 'q2', 'q3', 'q4', '1q', '2q', '3q', '4q',
            'eps', 'fiscal', 'fy25', 'fy26', 'quarterly',
            'annual report', 'beats estimate', 'misses estimate', 'earnings report',
            'earnings preview', 'earnings call', 'net income', 'gross margin',
            'operating income', 'ebitda', 'beats expectations', 'misses expectations'
        ],
        'Mergers & Acquisitions': [
            'merger', 'acquisition', 'acquire', 'buyout', 'takeover',
            'bought', 'purchase', 'acquiring', 'merge', 'consolidation',
            'joint venture', 'spinoff', 'spin-off',
            'demerger', 'divestiture', 'sale of', 'sells stake'
        ],
        'Guidance & Forecasts': [
            'guidance', 'forecast', 'outlook', 'projection', 'expects',
            'anticipates', 'target', 'price target', 'pt ', 'pt:', 'estimate',
            'analyst', 'upgrade', 'downgrade', 'rating', 'initiates coverage',
            'raises', 'lowers', 'reaffirms', 'maintains', 'sees', 'guides',
            '2025e', '2026e', '2027e', 'forward', 'next quarter'
        ],
        'Regulatory & Legal': [
            'lawsuit', 'litigation', 'legal', 'court', 'judge', 'ruling',
            'regulation', 'regulatory', 'sec ', 'ftc', 'doj', 'fda', 'fcc',
            'antitrust', 'investigation', 'probe', 'fine', 'penalty', 'settlement',
            'compliance', 'violation', 'sanction', 'policy', 'law', 'ban',
            'approved by', 'approval', 'denied', 'patent', 'copyright'
        ],
        'Product Launch': [
            'launches', 'unveiled', 'announces new', 'new product', 'new feature',
            'released', 'rolls out', 'introducing', 'debut', 'new service',
            'new model', 'version', 'update', 'upgrade to', 'expands distribution',
            'now available', 'coming soon', 'pre-order',
            'delayed', 'postponed', 'canceled', 'cancelled', 'scrapped', 'shelved',
            'foldable', 'launch plans'
        ],
        'Partnerships & Deals': [
            'partnership', 'partners with', 'collaboration', 'deal with',
            'agreement with', 'contract', 'signs', 'expands at', 'distribution',
            'teams up', 'joins forces', 'alliance', 'works with',
            'in talks', 'negotiating', 'discussing', 'considering', 'exploring',
            'cloud deal', 'strategic deal', 'interest from', 'interested parties'
        ],
        'Market Data': [
            'market cap', 'stock price', 'share price', 'trading at', 'trillion',
            'valuation', 'shares', 'stock is moving', 'from $', 'to $',
            'historical', 'all-time', 'since', 'million to', 'billion to'
        ],
        'Company Strategy': [
            'plans to', 'strategy', 'expansion', 'expanding', 'automation',
            'automate', 'operations', 'restructuring', 'reorganization',
            'scaling', 'growth plan', 'investing in', 'focuses on',
            'pivoting', 'shift', 'transformation', 'initiative',
            'robots', 'replace', 'workforce', 'ai-powered', 'digital transformation',
            'up for sale', 'received interest', 'snag the', 'potentially',
            'reinvent', 'effort to', 'development', 'breakthrough'
        ],
        'Company Metrics': [
            'million users', 'billion users', 'customers', 'subscribers',
            'hosted on', 'operates', 'manages', 'revenue of', 'sales of',
            'grew by', 'increased by', 'decreased by', 'user growth',
            'active users', 'monthly active', 'daily active'
        ],
        'Personnel Changes': [
            'ceo', 'chief executive', 'chief financial', 'cfo', 'cto', 'coo',
            'executive', 'appoints', 'hired', 'hires', 'joins as',
            'steps down', 'resigns', 'departing', 'layoffs', 'cuts',
            'fires', 'replaces', 'names', 'promoted to'
        ],
        'Breaking News': [
            'breaking', 'alert', 'just in', 'developing', 'urgent',
            'shutdown', 'crisis', 'crash', 'surge', 'plunge', 'spike',
            'record high', 'record low', 'halt', 'suspended',
            'emergency', 'unprecedented'
        ]
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

        return 'Other'


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
        clean = re.sub(r'\{Embed\}', '', text)
        clean = re.sub(r'\{Attachments\}', '', clean)
        clean = re.sub(r'https?://[^\s]+', '', clean)
        clean = re.sub(r'TweetShift[^\n]*', '', clean)
        clean = re.sub(r'Powered by [^\n]+', '', clean)
        clean = re.sub(r'📷\d+', '', clean)
        clean = re.sub(r'\[Tweeted\][^\n]*', '', clean)

        # Clean up whitespace
        clean = '\n'.join(line.strip() for line in clean.split('\n') if line.strip())
        clean = clean.strip()

        # Remove escape characters
        clean = re.sub(r'\\([()[\]{}.,!?\-\'#+_])', r'\1', clean)

        # Replace newlines with spaces for CSV
        clean = ' '.join(clean.split('\n'))

        return clean

    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract ticker symbols from text.

        Args:
            text: Message text

        Returns:
            List of ticker symbols
        """
        return re.findall(r'\$([A-Z][A-Z0-9]*)', text)

    def extract_tweet_url(self, text: str) -> str:
        """
        Extract tweet URL from message.

        Args:
            text: Message text

        Returns:
            Tweet URL or empty string
        """
        match = re.search(r'(https?://(?:twitter\.com|x\.com)/[^\s)]+)', text)
        return match.group(1) if match else ''

    def is_valid(self, text: str) -> bool:
        """
        Check if text meets minimum quality standards.

        Args:
            text: Cleaned text

        Returns:
            True if text is valid
        """
        return bool(text and len(text) >= self.min_text_length)


class DiscordParser:
    """Parses Discord channel export files."""

    TIMESTAMP_PATTERN = re.compile(r'\[(\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2})\]\s+(.*)')

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

        with open(file_path, 'r', encoding='utf-8') as f:
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
                    'timestamp': timestamp_match.group(1),
                    'username': timestamp_match.group(2).strip(),
                    'text_lines': []
                }
            elif current_message:
                current_message['text_lines'].append(line)

        # Process last message
        if current_message:
            messages.extend(self._process_message(current_message))

        return messages

    def _find_start_index(self, lines: List[str]) -> int:
        """Find the index where messages start."""
        for i, line in enumerate(lines):
            if line.strip().startswith('['):
                return i
        return 0

    def _process_message(self, message: Dict) -> List[Dict[str, str]]:
        """Process a single message and extract data."""
        full_text = ''.join(message['text_lines']).strip()

        # Extract components
        tickers = self.processor.extract_tickers(full_text)
        tweet_url = self.processor.extract_tweet_url(full_text)
        clean_text = self.processor.clean_text(full_text)

        # Validate
        if not self.processor.is_valid(clean_text):
            return []

        # Categorize
        category = self.categorizer.categorize(clean_text)

        # Create result entries
        result = []
        if tickers:
            for ticker in tickers:
                if ticker:
                    result.append({
                        'timestamp': message['timestamp'],
                        'author': message['username'],
                        'ticker': ticker,
                        'tweet_url': tweet_url,
                        'category': category,
                        'text': clean_text
                    })

        return result


class CSVWriter:
    """Writes messages to CSV with optional filtering and deduplication."""

    FIELDNAMES = ['timestamp', 'author', 'ticker', 'tweet_url', 'category', 'text']

    def __init__(self, deduplicate: bool = True, ticker_filter: Optional[Set[str]] = None):
        """
        Initialize CSV writer.

        Args:
            deduplicate: Remove duplicate messages
            ticker_filter: Optional set of allowed tickers
        """
        self.deduplicate = deduplicate
        self.ticker_filter = ticker_filter

    def write(self, messages: List[Dict[str, str]], output_path: Path) -> Dict[str, int]:
        """
        Write messages to CSV file.

        Args:
            messages: List of message dictionaries
            output_path: Output CSV file path

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_input': len(messages),
            'filtered': 0,
            'duplicates': 0,
            'written': 0
        }

        seen_messages = set()

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

            for msg in messages:
                # Apply ticker filter
                if self.ticker_filter and msg['ticker'].upper() not in self.ticker_filter:
                    stats['filtered'] += 1
                    continue

                # Check for duplicates
                if self.deduplicate:
                    unique_key = (msg['timestamp'], msg['ticker'].upper(), msg['text'])
                    if unique_key in seen_messages:
                        stats['duplicates'] += 1
                        continue
                    seen_messages.add(unique_key)

                # Write row
                writer.writerow({
                    'timestamp': self._convert_timestamp(msg['timestamp']),
                    'author': msg['author'],
                    'ticker': msg['ticker'],
                    'tweet_url': msg['tweet_url'],
                    'category': msg['category'],
                    'text': msg['text']
                })
                stats['written'] += 1

        return stats

    def _convert_timestamp(self, timestamp_str: str) -> str:
        """
        Convert timestamp from Jerusalem time to US Eastern time.
        
        Handles daylight saving time (DST) transitions correctly for both timezones.
        
        Args:
            timestamp_str: Timestamp in format 'DD/MM/YYYY HH:MM' (Jerusalem time)
            
        Returns:
            Timestamp in ISO format 'YYYY-MM-DD HH:MM:SS' (US Eastern time)
        """
        try:
            # Parse timestamp as naive datetime
            dt_naive = datetime.strptime(timestamp_str, '%d/%m/%Y %H:%M')
            
            # Define timezones
            jerusalem_tz = pytz.timezone('Asia/Jerusalem')
            eastern_tz = pytz.timezone('America/New_York')
            
            # Localize to Jerusalem time (this handles IST/IDT automatically)
            dt_jerusalem = jerusalem_tz.localize(dt_naive)
            
            # Convert to US Eastern time (handles EST/EDT automatically)
            dt_eastern = dt_jerusalem.astimezone(eastern_tz)
            
            # Return in ISO format
            return dt_eastern.strftime('%Y-%m-%d %H:%M:%S')
            
        except ValueError:
            return timestamp_str


class TickerFilter:
    """Loads and manages ticker filter lists."""

    @staticmethod
    def load_from_csv(file_path: Path) -> Set[str]:
        """
        Load ticker symbols from CSV file.

        Args:
            file_path: Path to CSV file with 'symbol' column

        Returns:
            Set of ticker symbols (uppercase)
        """
        tickers = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get('symbol', '').strip().upper()
                if ticker:
                    tickers.add(ticker)
        return tickers


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

    def convert(self,
                input_file: Path,
                output_file: Path,
                ticker_filter_file: Optional[Path] = None,
                verbose: bool = True) -> Dict[str, int]:
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

    def _print_stats(self, stats: Dict[str, int], messages: List[Dict[str, str]]):
        """Print conversion statistics."""
        print(f"\nConversion complete!")

        if stats['filtered'] > 0:
            print(f"Filtered out {stats['filtered']} messages (tickers not in filter list)")

        if stats['duplicates'] > 0:
            print(f"Removed {stats['duplicates']} duplicate messages")

        print(f"Written {stats['written']} unique messages to CSV")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert Discord channel exports to structured CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python discord_to_csv.py -i discord_data.txt -o output.csv

  # With ticker filter
  python discord_to_csv.py -i discord_data.txt -o output.csv -f tickers.csv

  # Disable deduplication
  python discord_to_csv.py -i discord_data.txt -o output.csv --no-dedup

  # Custom minimum text length
  python discord_to_csv.py -i discord_data.txt -o output.csv --min-length 100
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Input Discord export file'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output CSV file'
    )

    parser.add_argument(
        '-f', '--filter',
        type=Path,
        help='Ticker filter CSV file (with "symbol" column)'
    )

    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable duplicate message removal'
    )

    parser.add_argument(
        '--min-length',
        type=int,
        default=60,
        help='Minimum text length to keep (default: 60)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate filter file if provided
    if args.filter and not args.filter.exists():
        print(f"Error: Filter file not found: {args.filter}", file=sys.stderr)
        sys.exit(1)

    # Create converter and run
    try:
        converter = DiscordToCSVConverter(
            min_text_length=args.min_length,
            deduplicate=not args.no_dedup
        )

        stats = converter.convert(
            input_file=args.input,
            output_file=args.output,
            ticker_filter_file=args.filter,
            verbose=not args.quiet
        )

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
