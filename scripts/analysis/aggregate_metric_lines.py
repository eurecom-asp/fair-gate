#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def parse_metric_line(line: str):
    if not line.startswith("[METRIC]"):
        return None
    row = {}
    for token in line.strip().split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            row[key] = float(value)
        except ValueError:
            row[key] = value
    return row

def main():
    parser = argparse.ArgumentParser(description="Aggregate [METRIC] lines from evaluation logs.")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    rows = []
    for log_file in Path(args.log_dir).rglob("*.log"):
        with open(log_file, "r", errors="ignore") as f:
            for line in f:
                row = parse_metric_line(line)
                if row is not None:
                    row["log_file"] = str(log_file)
                    rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
