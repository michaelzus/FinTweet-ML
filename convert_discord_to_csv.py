"""Script to convert Discord data to CSV format with timestamp, ticker, and tweet text."""

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional


def parse_discord_file(file_path: Path) -> List[Dict[str, str]]:
    """
    Parse Discord export file and extract messages with timestamps and tickers.

    Args:
        file_path: Path to the Discord export text file

    Returns:
        List of dictionaries containing timestamp, ticker, and text
    """
    messages = []
    current_message = None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip header lines
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('['):
            start_idx = i
            break

    i = start_idx
    while i < len(lines):
        line = lines[i]

        # Check if line starts with timestamp pattern [DD/MM/YYYY HH:MM]
        timestamp_match = re.match(r'\[(\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2})\]\s+(.*)', line)

        if timestamp_match:
            # Save previous message if exists
            if current_message:
                messages.extend(process_message(current_message))

            # Start new message
            timestamp_str = timestamp_match.group(1)
            username = timestamp_match.group(2).strip()
            current_message = {
                'timestamp': timestamp_str,
                'username': username,
                'text_lines': []
            }
        elif current_message:
            # Add line to current message
            current_message['text_lines'].append(line)

        i += 1

    # Process last message
    if current_message:
        messages.extend(process_message(current_message))

    return messages


def categorize_message(text: str) -> str:
    """
    Categorize message into one of 12 major categories based on content.

    Args:
        text: Message text to categorize

    Returns:
        Category name
    """
    text_lower = text.lower()

    # Earnings - quarterly/annual reports, earnings surprises, outlooks
    earnings_keywords = [
        'earnings', 'quarter', 'q1', 'q2', 'q3', 'q4', '1q', '2q', '3q', '4q',
        'eps', 'fiscal', 'fy25', 'fy26', 'quarterly',
        'annual report', 'beats estimate', 'misses estimate', 'earnings report',
        'earnings preview', 'earnings call', 'net income', 'gross margin',
        'operating income', 'ebitda', 'beats expectations', 'misses expectations'
    ]
    if any(keyword in text_lower for keyword in earnings_keywords):
        return 'Earnings'

    # Mergers & Acquisitions - buyouts, mergers, strategic deals
    ma_keywords = [
        'merger', 'acquisition', 'acquire', 'buyout', 'takeover',
        'bought', 'purchase', 'acquiring', 'merge', 'consolidation',
        'joint venture', 'spinoff', 'spin-off',
        'demerger', 'divestiture', 'sale of', 'sells stake'
    ]
    if any(keyword in text_lower for keyword in ma_keywords):
        return 'Mergers & Acquisitions'

    # Guidance & Forecasts - forward-looking statements, projections
    guidance_keywords = [
        'guidance', 'forecast', 'outlook', 'projection', 'expects',
        'anticipates', 'target', 'price target', 'pt ', 'pt:', 'estimate',
        'analyst', 'upgrade', 'downgrade', 'rating', 'initiates coverage',
        'raises', 'lowers', 'reaffirms', 'maintains', 'sees', 'guides',
        '2025e', '2026e', '2027e', 'forward', 'next quarter'
    ]
    if any(keyword in text_lower for keyword in guidance_keywords):
        return 'Guidance & Forecasts'

    # Regulatory & Legal - lawsuits, government actions, policy changes
    regulatory_keywords = [
        'lawsuit', 'litigation', 'legal', 'court', 'judge', 'ruling',
        'regulation', 'regulatory', 'sec ', 'ftc', 'doj', 'fda', 'fcc',
        'antitrust', 'investigation', 'probe', 'fine', 'penalty', 'settlement',
        'compliance', 'violation', 'sanction', 'policy', 'law', 'ban',
        'approved by', 'approval', 'denied', 'patent', 'copyright'
    ]
    if any(keyword in text_lower for keyword in regulatory_keywords):
        return 'Regulatory & Legal'

    # Product Launch - new products, features, services, delays, cancellations
    product_keywords = [
        'launches', 'unveiled', 'announces new', 'new product', 'new feature',
        'released', 'rolls out', 'introducing', 'debut', 'new service',
        'new model', 'version', 'update', 'upgrade to', 'expands distribution',
        'now available', 'coming soon', 'pre-order',
        'delayed', 'postponed', 'canceled', 'cancelled', 'scrapped', 'shelved',
        'foldable', 'launch plans'
    ]
    if any(keyword in text_lower for keyword in product_keywords):
        return 'Product Launch'

    # Partnerships & Deals - strategic partnerships, collaborations, contracts
    partnership_keywords = [
        'partnership', 'partners with', 'collaboration', 'deal with',
        'agreement with', 'contract', 'signs', 'expands at', 'distribution',
        'teams up', 'joins forces', 'alliance', 'works with',
        'in talks', 'negotiating', 'discussing', 'considering', 'exploring',
        'cloud deal', 'strategic deal', 'interest from', 'interested parties'
    ]
    if any(keyword in text_lower for keyword in partnership_keywords):
        return 'Partnerships & Deals'

    # Market Data - stock performance, market cap, historical data, price movements
    market_data_keywords = [
        'market cap', 'stock price', 'share price', 'trading at', 'trillion',
        'valuation', 'shares', 'stock is moving', 'from $', 'to $',
        'historical', 'all-time', 'since', 'million to', 'billion to'
    ]
    if any(keyword in text_lower for keyword in market_data_keywords):
        return 'Market Data'

    # Company Strategy - business operations, expansion, automation, restructuring
    strategy_keywords = [
        'plans to', 'strategy', 'expansion', 'expanding', 'automation',
        'automate', 'operations', 'restructuring', 'reorganization',
        'scaling', 'growth plan', 'investing in', 'focuses on',
        'pivoting', 'shift', 'transformation', 'initiative',
        'robots', 'replace', 'workforce', 'ai-powered', 'digital transformation',
        'up for sale', 'received interest', 'snag the', 'potentially',
        'reinvent', 'effort to', 'development', 'breakthrough'
    ]
    if any(keyword in text_lower for keyword in strategy_keywords):
        return 'Company Strategy'

    # Company Metrics - revenue, users, growth metrics, operational stats
    metrics_keywords = [
        'million users', 'billion users', 'customers', 'subscribers',
        'hosted on', 'operates', 'manages', 'revenue of', 'sales of',
        'grew by', 'increased by', 'decreased by', 'user growth',
        'active users', 'monthly active', 'daily active'
    ]
    if any(keyword in text_lower for keyword in metrics_keywords):
        return 'Company Metrics'

    # Personnel Changes - executive moves, hirings, layoffs
    personnel_keywords = [
        'ceo', 'chief executive', 'chief financial', 'cfo', 'cto', 'coo',
        'executive', 'appoints', 'hired', 'hires', 'joins as',
        'steps down', 'resigns', 'departing', 'layoffs', 'cuts',
        'fires', 'replaces', 'names', 'promoted to'
    ]
    if any(keyword in text_lower for keyword in personnel_keywords):
        return 'Personnel Changes'

    # Breaking News - major unexpected market-moving events
    breaking_keywords = [
        'breaking', 'alert', 'just in', 'developing', 'urgent',
        'shutdown', 'crisis', 'crash', 'surge', 'plunge', 'spike',
        'record high', 'record low', 'halt', 'suspended',
        'emergency', 'unprecedented'
    ]
    if any(keyword in text_lower for keyword in breaking_keywords):
        return 'Breaking News'

    # Default to Other
    return 'Other'


def process_message(message: Dict) -> List[Dict[str, str]]:
    """
    Process a single message and extract tickers and text.

    Args:
        message: Dictionary containing timestamp, username, and text_lines

    Returns:
        List of dictionaries with timestamp, author, ticker, tweet_url, text, and category for each ticker found
    """
    # Combine all text lines
    full_text = ''.join(message['text_lines']).strip()

    # Find all tickers (words starting with $)
    tickers = re.findall(r'\$([A-Z][A-Z0-9]*)', full_text)

    # Extract tweet URL (twitter.com or x.com)
    tweet_url_match = re.search(r'(https?://(?:twitter\.com|x\.com)/[^\s)]+)', full_text)
    tweet_url = tweet_url_match.group(1) if tweet_url_match else ''

    # Remove common noise patterns
    clean_text = re.sub(r'\{Embed\}', '', full_text)
    clean_text = re.sub(r'\{Attachments\}', '', full_text)
    clean_text = re.sub(r'https?://[^\s]+', '', clean_text)
    clean_text = re.sub(r'TweetShift[^\n]*', '', clean_text)
    clean_text = re.sub(r'Powered by [^\n]+', '', clean_text)
    clean_text = re.sub(r'📷\d+', '', clean_text)
    clean_text = re.sub(r'\[Tweeted\][^\n]*', '', clean_text)
    clean_text = '\n'.join(line.strip() for line in clean_text.split('\n') if line.strip())
    clean_text = clean_text.strip()

    # Remove escape characters (backslashes before special chars)
    clean_text = re.sub(r'\\([()[\]{}.,!?\-\'#+_])', r'\1', clean_text)

    # Replace newlines with spaces to keep CSV on single lines
    clean_text = ' '.join(clean_text.split('\n'))

    # If no text after cleaning or text is too short, skip
    if not clean_text or len(clean_text) < 60:
        return []

    # Categorize the message
    category = categorize_message(clean_text)

    result = []

    if tickers:
        # Create entry for each ticker found (only if both ticker and text exist)
        for ticker in tickers:
            if ticker and clean_text:  # Both must be non-empty
                result.append({
                    'timestamp': message['timestamp'],
                    'author': message['username'],
                    'ticker': ticker,
                    'tweet_url': tweet_url,
                    'category': category,
                    'text': clean_text
                })
    # Skip messages without tickers (filter out empty ticker rows)

    return result


def convert_timestamp(timestamp_str: str) -> str:
    """
    Convert timestamp from DD/MM/YYYY HH:MM to ISO format.

    Args:
        timestamp_str: Timestamp string in DD/MM/YYYY HH:MM format

    Returns:
        Timestamp in ISO format (YYYY-MM-DD HH:MM:SS)
    """
    try:
        dt = datetime.strptime(timestamp_str, '%d/%m/%Y %H:%M')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return timestamp_str


def load_ticker_filter(filter_file: Path) -> Set[str]:
    """
    Load ticker symbols from filter file.

    Args:
        filter_file: Path to CSV file containing ticker symbols

    Returns:
        Set of ticker symbols (uppercase)
    """
    tickers = set()
    with open(filter_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get('symbol', '').strip().upper()
            if ticker:
                tickers.add(ticker)
    return tickers


def write_to_csv(messages: List[Dict[str, str]], output_path: Path, ticker_filter: Optional[Set[str]] = None) -> None:
    """
    Write messages to CSV file with deduplication.

    Args:
        messages: List of message dictionaries
        output_path: Path to output CSV file
        ticker_filter: Optional set of ticker symbols to filter by
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'author', 'ticker', 'tweet_url', 'category', 'text'])
        writer.writeheader()

        filtered_count = 0
        written_count = 0
        duplicate_count = 0

        # Track unique messages to detect duplicates
        seen_messages = set()

        for msg in messages:
            # Apply ticker filter if provided
            if ticker_filter and msg['ticker'].upper() not in ticker_filter:
                filtered_count += 1
                continue

            # Create unique key: timestamp + ticker + text (excluding minor variations)
            unique_key = (msg['timestamp'], msg['ticker'].upper(), msg['text'])

            # Skip if we've already written this exact message
            if unique_key in seen_messages:
                duplicate_count += 1
                continue

            seen_messages.add(unique_key)

            writer.writerow({
                'timestamp': convert_timestamp(msg['timestamp']),
                'author': msg['author'],
                'ticker': msg['ticker'],
                'tweet_url': msg['tweet_url'],
                'category': msg['category'],
                'text': msg['text']
            })
            written_count += 1

        if ticker_filter:
            print(f"\nFiltered out {filtered_count} messages (tickers not in filter list)")

        if duplicate_count > 0:
            print(f"Removed {duplicate_count} duplicate messages")

        print(f"Written {written_count} unique messages to CSV")


def main(use_ticker_filter: bool = True) -> None:
    """
    Main function to convert Discord data to CSV.

    Args:
        use_ticker_filter: If True, filter tickers using high_volume.csv
    """
    # Input file
    discord_file = Path(__file__).parent / 'discrod_data' / 'AI.INVEST.ISRAEL - Text Channels - עדכוני-שוק-ההון💰 [1430056629554122774].txt'

    # Output file
    output_file = Path(__file__).parent / 'output' / 'discord_messages.csv'
    output_file.parent.mkdir(exist_ok=True)

    # Ticker filter file
    ticker_filter_file = Path(__file__).parent / 'high_volume.csv'

    print(f"Reading Discord data from: {discord_file}")
    messages = parse_discord_file(discord_file)

    print(f"Found {len(messages)} messages")

    # Load ticker filter if requested
    ticker_filter = None
    if use_ticker_filter and ticker_filter_file.exists():
        ticker_filter = load_ticker_filter(ticker_filter_file)
        print(f"Loaded {len(ticker_filter)} tickers from filter list: {ticker_filter_file}")
    elif use_ticker_filter:
        print(f"Warning: Ticker filter file not found: {ticker_filter_file}")
        print("Proceeding without ticker filter")

    print(f"Writing to CSV: {output_file}")
    write_to_csv(messages, output_file, ticker_filter)

    print("Conversion complete!")

    # Count messages that match filter
    if ticker_filter:
        filtered_messages = [msg for msg in messages if msg['ticker'].upper() in ticker_filter]
        print(f"\nTotal messages after filtering: {len(filtered_messages)}")
        messages_for_stats = filtered_messages
    else:
        messages_for_stats = messages

    # Print sample of results
    if messages_for_stats:
        print("\nSample of first 5 entries:")
        for i, msg in enumerate(messages_for_stats[:5]):
            print(f"\n{i+1}. Timestamp: {msg['timestamp']}")
            print(f"   Author: {msg['author']}")
            print(f"   Ticker: {msg['ticker'] or '(none)'}")
            print(f"   Tweet URL: {msg['tweet_url'] or '(none)'}")
            print(f"   Category: {msg['category']}")
            print(f"   Text: {msg['text'][:100]}...")

        # Print category distribution
        from collections import Counter
        category_counts = Counter(msg['category'] for msg in messages_for_stats)
        print("\n\nCategory Distribution:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(messages_for_stats)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")


if __name__ == '__main__':
    main()
