"""Script to load and display the last 20 rows from a feather file."""

import pandas as pd


def main() -> None:
    """Load MDB.feather file and print the last 20 lines."""
    file_path = "data/daily/MDB.feather"
    
    # Load the feather file
    df = pd.read_feather(file_path)
    
    # Print basic info
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nLast 20 rows:")
    print("-" * 80)
    
    # Print last 20 rows
    print(df.tail(20).to_string())


if __name__ == "__main__":
    main()




