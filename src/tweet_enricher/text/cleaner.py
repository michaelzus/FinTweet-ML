"""Text cleaning and normalization for FinBERT processing.

This module provides text cleaning optimized for FinBERT sentiment analysis:
- Unicode normalization (NFKC) for fancy fonts
- Emoji-to-text mapping for sentiment preservation
- Punctuation normalization
- Removal of zero-width and invisible characters
"""

import re
import unicodedata
from typing import Dict

# =============================================================================
# Finance-specific emoji mappings
# Based on actual data analysis of 34,899 tweets with 49,044 emoji occurrences
# =============================================================================

FINANCE_EMOJI_MAP: Dict[str, str] = {
    # Market direction indicators (high frequency)
    "🟢": "[green]",  # 11,879x - bullish/up signal
    "🔴": "[red]",  # 3,919x - bearish/down signal
    "🟡": "[yellow]",  # 647x - neutral/caution
    "📈": "[up]",  # 209x - uptrend
    "📉": "[down]",  # 126x - downtrend
    "🔼": "[up]",  # 6x - up triangle
    "🔽": "[down]",  # 6x - down triangle
    "🔻": "[down]",  # 17x - down red triangle
    "🟠": "[orange]",  # 7x - caution
    "🔵": "[blue]",  # 7x - neutral
    # Sentiment signals (valuable for FinBERT)
    "🚀": "[rocket]",  # 251x - bullish momentum
    "🔥": "[fire]",  # 291x - hot/trending
    "💩": "[bad]",  # 97x - negative sentiment
    "😭": "[crying]",  # 73x - panic/despair
    "😂": "[laughing]",  # 198x - mockery/disbelief
    "🤣": "[laughing]",  # 49x - same
    "😐": "[neutral]",  # 59x - flat/unchanged
    "🥶": "[cold]",  # 115x - frozen/dead stock
    "💀": "[dead]",  # 18x - death cross/terrible
    "✅": "[check]",  # 178x - confirmed/positive
    "❌": "[x]",  # 54x - negative/denied
    "👀": "[eyes]",  # 138x - watching/attention
    "🤔": "[thinking]",  # 28x - uncertain
    "🙌": "[celebrate]",  # 49x - bullish celebration
    "🙏": "[hope]",  # 45x - hoping/praying
    "👑": "[king]",  # 60x - best performer
    "🏆": "[trophy]",  # 9x - winner
    "🎯": "[target]",  # 17x - price target hit
    "🤯": "[shocked]",  # 33x - surprised
    "😳": "[surprised]",  # 44x - flushed/shocked
    "😑": "[flat]",  # 26x - expressionless
    "😕": "[confused]",  # 25x - uncertain
    "😮": "[wow]",  # 9x - surprised
    "😲": "[shocked]",  # 4x - astonished
    "😅": "[nervous]",  # 4x - nervous laugh
    "😉": "[wink]",  # 16x - confident
    "😎": "[cool]",  # 5x - confident
    "🫡": "[salute]",  # 12x - respect
    "🤝": "[handshake]",  # 19x - deal/agreement
    "💪": "[strong]",  # 1x - strength
    "❤": "[heart]",  # 16x - love
    "💚": "[green_heart]",  # 6x - bullish love
    "🥲": "[bittersweet]",  # 5x - mixed feelings
    "🥹": "[emotional]",  # 4x - holding back tears
    "🫣": "[peeking]",  # 6x - cautious look
    "🤷": "[shrug]",  # 4x - uncertain
    # Finance-specific context
    "📊": "[chart]",  # 562x - data/analysis
    "🏦": "[bank]",  # 44x - banking sector
    "💰": "[money]",  # 55x - money/profit
    "💸": "[money_fly]",  # 17x - money leaving
    "💳": "[card]",  # 55x - credit/payment
    "💼": "[business]",  # 13x - corporate
    "🪙": "[coin]",  # 12x - crypto
    "💵": "[dollar]",  # 2x - cash
    "💹": "[chart_yen]",  # 5x - markets
    "💯": "[hundred]",  # 2x - perfect
    "♾": "[infinity]",  # 12x - unlimited
    # Sector indicators
    "💊": "[pharma]",  # 44x - healthcare/pharma
    "💉": "[vaccine]",  # 7x - healthcare
    "🦠": "[virus]",  # 6x - biotech
    "🧫": "[biotech]",  # 2x - lab
    "🏥": "[hospital]",  # 6x - healthcare
    "⚕": "[medical]",  # 3x - healthcare
    "🤖": "[ai]",  # 33x - AI/tech
    "📱": "[mobile]",  # 83x - tech/mobile
    "💻": "[computer]",  # 51x - tech
    "🖥": "[desktop]",  # 36x - tech
    "☁": "[cloud]",  # 86x - cloud computing
    "⚡": "[energy]",  # 35x - energy/power
    "🛢": "[oil]",  # 8x - oil/energy
    "🚗": "[auto]",  # 13x - automotive
    "🚖": "[taxi]",  # 14x - ride-sharing
    "🚘": "[car]",  # 1x - automotive
    "🛩": "[airline]",  # 19x - aviation
    "✈": "[plane]",  # airline
    "🚢": "[ship]",  # 1x - shipping
    "🛳": "[cruise]",  # 6x - cruise lines
    "🚚": "[truck]",  # 8x - logistics
    "🏈": "[football]",  # 17x - sports betting
    "🎮": "[gaming]",  # 23x - gaming sector
    "🕹": "[joystick]",  # 10x - gaming
    "🍏": "[apple]",  # 15x - AAPL
    "🍎": "[apple_red]",  # 15x - AAPL
    "🐶": "[doge]",  # 20x - DOGE/meme stocks
    "🦅": "[eagle]",  # 5x - America/patriotic
    "🌊": "[wave]",  # 5x - momentum
    "🧠": "[brain]",  # 15x - AI/smart
    "🦾": "[robot_arm]",  # 14x - automation
    "🦿": "[robot_leg]",  # 1x - automation
    "🏢": "[office]",  # 14x - real estate
    "🏰": "[castle]",  # 13x - fortress
    "🏭": "[factory]",  # 4x - manufacturing
    "🏗": "[construction]",  # 9x - building
    "👷": "[worker]",  # 6x - labor
    "🛒": "[shopping]",  # 23x - retail/e-commerce
    "🛍": "[bags]",  # 12x - retail
    "📦": "[package]",  # 31x - delivery/logistics
    "📺": "[tv]",  # 35x - media
    "🎬": "[movie]",  # 8x - entertainment
    "🎥": "[camera]",  # 3x - media
    "🎧": "[headphones]",  # 11x - audio/streaming
    "🎤": "[mic]",  # 4x - media
    "🎙": "[podcast]",  # 1x - media
    "🌐": "[globe]",  # 49x - global/international
    "🌍": "[earth]",  # 3x - global
    "🌎": "[americas]",  # 3x - US markets
    "🌏": "[asia]",  # 6x - Asian markets
    "🍕": "[pizza]",  # 15x - food sector
    "🍟": "[fries]",  # 8x - fast food
    "🍔": "[burger]",  # 4x - fast food
    "🌮": "[taco]",  # 3x - fast food
    "🌯": "[burrito]",  # 6x - food
    "🥤": "[drink]",  # 33x - beverages
    "☕": "[coffee]",  # 8x - coffee/Starbucks
    "🍺": "[beer]",  # 10x - alcohol
    "🍿": "[popcorn]",  # 20x - entertainment
    "🍪": "[cookie]",  # 4x - food
    "🥕": "[carrot]",  # 7x - food/health
    "🌿": "[herb]",  # 8x - cannabis?
    "🍄": "[mushroom]",  # 2x - psychedelics?
    "🌱": "[seedling]",  # 6x - growth/ESG
    "🌟": "[star]",  # 27x - highlight
    "⭐": "[star]",  # star
    "🔬": "[research]",  # 12x - R&D
    "🔐": "[secure]",  # 13x - security
    "🔒": "[lock]",  # 12x - security
    "🔑": "[key]",  # key
    "📶": "[signal]",  # 10x - telecom
    "📡": "[antenna]",  # 1x - telecom
    "🛰": "[satellite]",  # 4x - space
    "💡": "[idea]",  # 11x - innovation
    "⚙": "[gear]",  # 44x - settings/engineering
    "🛠": "[tools]",  # 9x - maintenance
    "⚖": "[scales]",  # 2x - legal/balance
    "🧴": "[lotion]",  # 12x - consumer goods
    "👟": "[shoe]",  # 18x - retail/Nike
    "👔": "[tie]",  # 14x - business
    "👜": "[handbag]",  # 6x - luxury
    "👓": "[glasses]",  # 3x - eyewear
    "🎿": "[ski]",  # 2x - leisure
    "⛷": "[skier]",  # 4x - leisure
    "🧘": "[yoga]",  # 18x - wellness
    "🛌": "[sleep]",  # 2x - rest
    "🔮": "[crystal]",  # 2x - prediction
    "🐐": "[goat]",  # 2x - greatest of all time
    "🦈": "[shark]",  # 1x - predator
    "🐳": "[whale]",  # 1x - big investor
    "🦚": "[peacock]",  # 3x - NBC?
    "🦎": "[lizard]",  # 4x - gecko?
    "🏝": "[island]",  # 4x - vacation
    "⛰": "[mountain]",  # 3x - obstacle
    "🌪": "[tornado]",  # 1x - chaos
    "🌀": "[cyclone]",  # 1x - chaos
    "❄": "[cold]",  # 18x - frozen
    "🧨": "[explosive]",  # 1x - volatile
    "💥": "[explosion]",  # 1x - breakout
    "💨": "[dash]",  # 3x - fast
    "🔊": "[loud]",  # 2x - announcement
    "🔋": "[battery]",  # 1x - energy storage
    "🛞": "[wheel]",  # 1x - automotive
    "🫒": "[olive]",  # 1x - food
    # Rankings/medals
    "🥇": "[first]",  # 337x - top performer
    "🥈": "[second]",  # 386x - second place
    "🥉": "[third]",  # 386x - third place
    # Formatting emojis - remove (no semantic value for FinBERT)
    "🔹": "",  # 10,789x - bullet decoration
    "🔸": "",  # 2,475x - bullet decoration
    "🔷": "",  # 20x - diamond decoration
    "📢": "",  # 5,459x - announcement marker
    "🚨": "[alert]",  # 2,733x - breaking news (keep as alert)
    "➤": "",  # 2,197x - arrow decoration
    "➡": "",  # 765x - arrow
    "👇": "",  # 1,874x - "see below"
    "👉": "",  # 433x - "see this"
    "👈": "",  # 2x - "see left"
    "🧵": "[thread]",  # 48x - thread marker
    "🔔": "",  # 72x - notification bell
    "🗓": "",  # 65x - calendar
    "📆": "",  # 12x - calendar
    "📝": "",  # 7x - memo
    "✍": "",  # 10x - writing
    "🔍": "",  # 9x - search
    "🔎": "",  # 1x - search
    "🔗": "",  # 1x - link
    "📌": "",  # 1x - pin
    "🎟": "",  # 1x - ticket
    "✔": "[check]",  # 7x - checkmark
    "🔘": "",  # 1x - radio button
    "🔃": "",  # 4x - refresh
    "🔄": "",  # 1x - refresh
    "🆕": "",  # 6x - new
    "🟥": "",  # 1x - red square
    "⬛": "",  # 1x - black square
    "▪": "",  # 97x - small square
    "▫": "",  # 19x - white square
    "◾": "",  # 6x - medium square
    "●": "",  # 1x - circle
    "⚫": "",  # 4x - black circle
    "🕵": "",  # 15x - spy
    "👨": "",  # 3x - man
    "👩": "",  # 3x - woman
    "👧": "",  # 3x - girl
    "👦": "",  # 3x - boy
    "🧑": "",  # 14x - adult
    "👐": "",  # 1x - open hands
    "🤳": "",  # 1x - selfie
    "🤦": "",  # 1x - facepalm
    "🤞": "",  # 1x - crossed fingers
    "🧐": "",  # 1x - monocle
    "🥽": "",  # 1x - goggles
    "😍": "",  # 1x - heart eyes
    "😋": "",  # 1x - yummy
    "😪": "",  # 2x - sleepy
    "😏": "",  # 2x - smirk
    "🥺": "",  # 1x - pleading
    "🏎": "",  # 3x - race car
    "💎": "[diamond]",  # 1x - diamond hands
    "♟": "",  # 1x - chess pawn
    "🗡": "",  # 1x - dagger
    # Variation selector - always remove (invisible modifier)
    "️": "",  # 1,489x - variation selector-16
    # Skin tone modifiers - remove
    "🏻": "",  # 18x - light skin
    "🏼": "",  # 1x - medium-light skin
    # Male/female signs
    "♂": "",  # 5x - male sign
}

# Regional indicator letters (flags decomposed) - all map to empty string
REGIONAL_INDICATORS: Dict[str, str] = {chr(c): "" for c in range(0x1F1E6, 0x1F200)}

# =============================================================================
# Punctuation normalization
# Smart quotes, dashes, bullets -> ASCII equivalents
# =============================================================================

PUNCTUATION_MAP: Dict[str, str] = {
    "'": "'",  # U+2019 RIGHT SINGLE QUOTATION MARK (13,128x)
    "'": "'",  # U+2018 LEFT SINGLE QUOTATION MARK (322x)
    """: '"',    # U+201C LEFT DOUBLE QUOTATION MARK (3,305x)
    """: '"',  # U+201D RIGHT DOUBLE QUOTATION MARK (3,292x)
    "—": " - ",  # U+2014 EM DASH (4,451x) - with spaces for readability
    "–": "-",  # U+2013 EN DASH (4,205x)
    "…": "...",  # U+2026 HORIZONTAL ELLIPSIS (1,488x)
    "•": " - ",  # U+2022 BULLET (2,425x)
    "‣": " - ",  # U+2023 TRIANGULAR BULLET (25x)
    "‑": "-",  # U+2011 NON-BREAKING HYPHEN (23x)
    "∙": "-",  # U+2219 BULLET OPERATOR (2x)
    "：": ":",  # U+FF1A FULLWIDTH COLON (4x)
}

# =============================================================================
# Arrow symbols -> directional tokens or removal
# =============================================================================

ARROW_MAP: Dict[str, str] = {
    "↑": "[up]",  # U+2191 UPWARDS ARROW (232x)
    "↓": "[down]",  # U+2193 DOWNWARDS ARROW (70x)
    "⬆": "[up]",  # U+2B06 UPWARDS BLACK ARROW (28x)
    "⬇": "[down]",  # U+2B07 DOWNWARDS BLACK ARROW (138x)
    "▲": "[up]",  # U+25B2 BLACK UP-POINTING TRIANGLE (72x)
    "▼": "[down]",  # U+25BC BLACK DOWN-POINTING TRIANGLE (24x)
    "↗": "[up]",  # U+2197 NORTH EAST ARROW (9x)
    "→": "",  # U+2192 RIGHTWARDS ARROW (134x) - formatting
    "▶": "",  # U+25B6 BLACK RIGHT-POINTING TRIANGLE (6x)
    "↔": "",  # U+2194 LEFT RIGHT ARROW (2x)
    "↳": "",  # U+21B3 DOWNWARDS ARROW WITH TIP RIGHTWARDS (1x)
    "⬅": "",  # U+2B05 LEFTWARDS BLACK ARROW (1x)
}

# =============================================================================
# Zero-width and invisible characters to remove
# =============================================================================

ZERO_WIDTH_CHARS: set[str] = {
    "\u200b",  # ZERO WIDTH SPACE (32x)
    "\u200c",  # ZERO WIDTH NON-JOINER (5x)
    "\u200d",  # ZERO WIDTH JOINER (31x)
    "\u2060",  # WORD JOINER (1x)
    "\ufffc",  # OBJECT REPLACEMENT CHARACTER (1x)
    "\u20e3",  # COMBINING ENCLOSING KEYCAP (3x)
}

# Whitespace variants to normalize to regular space
WHITESPACE_VARIANTS: Dict[str, str] = {
    "\u202f": " ",  # NARROW NO-BREAK SPACE (441x)
    "\u00a0": " ",  # NO-BREAK SPACE (54x)
    "\u2009": " ",  # THIN SPACE (4x)
    "\u2002": " ",  # EN SPACE (2x)
}

# =============================================================================
# Regex patterns
# =============================================================================

# Pattern for remaining emojis not in our mapping (to remove)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001f9ff"  # Misc Symbols, Emoticons, etc.
    "\U00002600-\U000027bf"  # Dingbats, Misc symbols
    "\U0001fa00-\U0001faff"  # Supplemental Symbols
    "\U0001f600-\U0001f64f"  # Emoticons
    "\U0001f1e0-\U0001f1ff"  # Regional indicators (flags)
    "\U0001f000-\U0001f02f"  # Mahjong, dominos
    "\U0000fe00-\U0000fe0f"  # Variation selectors
    "\U000e0000-\U000e007f"  # Tags
    "]+"
)

# Pattern to collapse multiple spaces
MULTI_SPACE_PATTERN = re.compile(r" {2,}")


def clean_for_finbert(text: str) -> str:
    """
    Clean and normalize text for FinBERT processing.

    Applies the following transformations in order:
    1. NFKC Unicode normalization (fixes fancy fonts)
    2. Remove zero-width characters
    3. Normalize whitespace variants
    4. Normalize punctuation (smart quotes, dashes)
    5. Map arrows to directional tokens
    6. Map finance emojis to semantic tokens
    7. Remove regional indicators (flag components)
    8. Remove remaining unmapped emojis
    9. Collapse multiple spaces

    Args:
        text: Raw text input

    Returns:
        Cleaned text optimized for FinBERT tokenization
    """
    if not text:
        return text

    # 1. NFKC normalization - converts mathematical bold, compatibility chars
    # This handles ~57k occurrences of fancy Twitter fonts
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove zero-width characters
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, "")

    # 3. Normalize whitespace variants to regular space
    for char, replacement in WHITESPACE_VARIANTS.items():
        text = text.replace(char, replacement)

    # 4. Normalize punctuation
    for char, replacement in PUNCTUATION_MAP.items():
        text = text.replace(char, replacement)

    # 5. Map arrows to directional tokens
    for char, replacement in ARROW_MAP.items():
        text = text.replace(char, replacement)

    # 6. Map finance emojis to semantic tokens
    for emoji, token in FINANCE_EMOJI_MAP.items():
        if emoji in text:
            # Add spaces around tokens to ensure proper tokenization
            replacement = f" {token} " if token else ""
            text = text.replace(emoji, replacement)

    # 7. Remove regional indicators (flag components)
    for char in REGIONAL_INDICATORS:
        text = text.replace(char, "")

    # 8. Remove any remaining unmapped emojis
    text = EMOJI_PATTERN.sub("", text)

    # 9. Collapse multiple spaces and strip
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = text.strip()

    return text
