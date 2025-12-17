#!/usr/bin/env python3
"""Minimal test to verify TwitterAPI.io response structure."""

import json

import requests
import urllib3

# Suppress SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API credentials
API_KEY = "new1_1c8b0739b163449aa67c665c484079a3"
BASE_URL = "https://api.twitterapi.io"


def test_single_tweet():
    """Fetch and display full structure of one tweet."""
    print("\n🔌 TwitterAPI.io - Single Tweet Test\n")
    print("=" * 70)

    headers = {"X-API-Key": API_KEY}
    url = f"{BASE_URL}/twitter/user/last_tweets"
    params = {"userName": "StockMKTNewz"}

    try:
        response = requests.get(url, headers=headers, params=params, verify=False, timeout=60)
        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"Error: {response.text}")
            return

        resp_data = response.json()

        # Show top-level structure
        print(f"\n📦 Response Structure:")
        print(f"   status: {resp_data.get('status')}")
        print(f"   code: {resp_data.get('code')}")
        print(f"   has_next_page: {resp_data.get('has_next_page')}")
        print(f"   next_cursor: {'Yes' if resp_data.get('next_cursor') else 'No'}")

        # Get tweets
        data = resp_data.get("data", {})
        tweets = data.get("tweets", [])
        print(f"\n📊 Tweets count: {len(tweets)}")

        if not tweets:
            print("No tweets found!")
            return

        # Show FULL structure of first tweet
        tweet = tweets[0]
        print("\n" + "=" * 70)
        print("🐦 FIRST TWEET - FULL STRUCTURE:")
        print("=" * 70)
        print(json.dumps(tweet, indent=2, ensure_ascii=False))

        # Highlight key fields for our use case
        print("\n" + "=" * 70)
        print("📋 KEY FIELDS FOR OUR USE CASE:")
        print("=" * 70)
        print(f"   ID:          {tweet.get('id')}")
        print(f"   URL:         {tweet.get('url')}")
        print(f"   Twitter URL: {tweet.get('twitterUrl')}")
        print(f"   Created At:  {tweet.get('createdAt')}")  # Check various field names
        print(f"   created_at:  {tweet.get('created_at')}")
        print(f"   Text:        {tweet.get('text', '')[:100]}...")

        # Check author info
        author = tweet.get("author", {})
        if author:
            print(f"\n   Author Fields:")
            print(f"      userName:    {author.get('userName')}")
            print(f"      name:        {author.get('name')}")
            print(f"      id:          {author.get('id')}")

        print("\n✅ Test complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    test_single_tweet()
