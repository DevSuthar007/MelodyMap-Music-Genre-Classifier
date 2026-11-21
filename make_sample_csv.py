"""Create a one-song CSV in the exact format the web app expects.

Usage:
    python src/make_sample_csv.py --data data/features_30_sec.csv --row 0
"""
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/features_30_sec.csv")
parser.add_argument("--row", type=int, default=0, help="which song (row index) to export")
parser.add_argument("--out", default="sample_input.csv")
args = parser.parse_args()

df = pd.read_csv(args.data)
sample = df.drop(columns=["filename", "length", "label"]).iloc[[args.row]]
sample.to_csv(args.out, index=False)
print(f"Saved {args.out} (actual genre: {df.loc[args.row, 'label']})")
