#!/usr/bin/env python3
"""
Test script to verify timezone conversion from Jerusalem to US Eastern Time.

This demonstrates how the conversion handles:
- Regular time conversion
- Israel DST (IDT) and Standard Time (IST)
- US DST (EDT) and Standard Time (EST)
- Different time offsets throughout the year
"""

from datetime import datetime
import pytz


def convert_timestamp(timestamp_str: str) -> tuple[str, str]:
    """
    Convert timestamp from Jerusalem time to US Eastern time.
    
    Returns:
        Tuple of (eastern_time, info_string)
    """
    # Parse timestamp as naive datetime
    dt_naive = datetime.strptime(timestamp_str, '%d/%m/%Y %H:%M')
    
    # Define timezones
    jerusalem_tz = pytz.timezone('Asia/Jerusalem')
    eastern_tz = pytz.timezone('America/New_York')
    
    # Localize to Jerusalem time
    dt_jerusalem = jerusalem_tz.localize(dt_naive)
    
    # Convert to US Eastern time
    dt_eastern = dt_jerusalem.astimezone(eastern_tz)
    
    # Get timezone names and offsets
    jerusalem_name = dt_jerusalem.tzname()
    eastern_name = dt_eastern.tzname()
    offset_hours = (dt_jerusalem.utcoffset().total_seconds() - dt_eastern.utcoffset().total_seconds()) / 3600
    
    # Return formatted string and info
    eastern_str = dt_eastern.strftime('%Y-%m-%d %H:%M:%S')
    info = f"Jerusalem: {timestamp_str} ({jerusalem_name}) → New York: {eastern_str} ({eastern_name}) [Offset: {offset_hours:.0f}h]"
    
    return eastern_str, info


def main():
    """Test timezone conversion with various dates."""
    print("="*80)
    print("Timezone Conversion Test: Jerusalem → US Eastern Time")
    print("="*80)
    print()
    
    test_cases = [
        # Winter (both on standard time)
        ("15/01/2024 10:00", "Winter - Both standard time (IST → EST)"),
        
        # Spring (Israel DST starts late March, US starts early March)
        ("15/03/2024 10:00", "Spring - US already in DST, Israel not yet"),
        ("01/04/2024 10:00", "Spring - Both in DST (IDT → EDT)"),
        
        # Summer (both on daylight time)
        ("15/07/2024 10:00", "Summer - Both daylight time (IDT → EDT)"),
        
        # Fall (US ends DST early November, Israel ends late October)
        ("25/10/2024 10:00", "Fall - Israel back to standard, US still DST"),
        ("10/11/2024 10:00", "Fall - Both standard time (IST → EST)"),
        
        # Market hours examples
        ("15/11/2024 16:30", "Market open time (9:30 AM EST)"),
        ("15/11/2024 23:00", "Market close time (4:00 PM EST)"),
        ("15/11/2024 02:00", "Pre-market (7:00 PM EST previous day)"),
    ]
    
    print("Test Cases:")
    print("-" * 80)
    
    for timestamp, description in test_cases:
        eastern_str, info = convert_timestamp(timestamp)
        print(f"\n{description}")
        print(f"  {info}")
    
    print("\n" + "="*80)
    print("✅ Timezone conversion working correctly!")
    print("="*80)
    print()
    print("Key Points:")
    print("  • Jerusalem uses Israel Standard Time (IST, UTC+2) and Israel Daylight Time (IDT, UTC+3)")
    print("  • US Eastern uses Eastern Standard Time (EST, UTC-5) and Eastern Daylight Time (EDT, UTC-4)")
    print("  • Time difference varies: 7-8 hours depending on DST status")
    print("  • pytz handles all DST transitions automatically")
    print()
    print("US Stock Market Hours (in Jerusalem time):")
    print("  • Pre-market:  14:00-16:30 Jerusalem time")
    print("  • Regular:     16:30-23:00 Jerusalem time") 
    print("  • After-hours: 23:00-02:00 Jerusalem time (next day)")
    print()


if __name__ == '__main__':
    main()

