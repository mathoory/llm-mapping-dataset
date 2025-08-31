# ---- CLI ----
"""
Usage:
    python src/utils/analyze.py path/to/results_dir --out_dir path/to/output_dir
Notes:
    - Reads every *.json file in results_dir.
    - Each file must contain a JSON array of result dicts (not JSONL).
"""
import os, json, glob
import pandas as pd

REQUIRED_COLS = {"TOPIC","MODEL","DIFFICULTY","ACC","ERRORS"}

def load_results_dir(results_dir):
    """Load and concatenate all JSON arrays from results_dir."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No .json files found in {results_dir}")
    all_rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{p} is not a JSON array.")
        all_rows.extend(data)
    df = pd.DataFrame(all_rows)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in merged data: {missing}")
    return df

## pretty_topic function removed; use pretty_topic_map directly
pretty_topic_map = {
    "lower to upper string": "Lowercase to Uppercase (random)",
    "lower to upper words": "Lowercase to Uppercase (words)",
    "lower to upper natural text": "Lowercase to Uppercase (text)",
    "upper to lower string": "Uppercase to Lowercase (random)",
    "upper to lower words": "Uppercase to Lowercase (words)",
    "upper to lower natural text": "Uppercase to Lowercase (text)",
    "RNA": "RNA to Protein",
    "digits": "Digits to Number Words",
    "numbers": "Number Words to Digits",
    "country code to country": "ISO-3166 Code to Country",
}

def build_topic_table(df_topic, decimals=2):
    # mean accuracy per difficulty per model
    acc = (
        df_topic.groupby(["MODEL", "DIFFICULTY"])["ACC"]
        .mean()
        .unstack("DIFFICULTY")
        .reindex(columns=["easy", "medium", "hard"])
    )
    acc["Average"] = acc.mean(axis=1)

    # aggregate error counts per model (warning-free)
    tmp = df_topic.assign(
        _ins=df_topic["ERRORS"].apply(lambda e: e.get("INSERTIONS", 0)),
        _del=df_topic["ERRORS"].apply(lambda e: e.get("DELETIONS", 0)),
        _sub=df_topic["ERRORS"].apply(lambda e: e.get("SUBSTITUTIONS", 0)),
    ).groupby("MODEL")[["_ins", "_del", "_sub"]].sum()

    denom = tmp.sum(axis=1).replace(0, 1)  # avoid /0
    shares = (tmp.div(denom, axis=0) * 100).round(1)

    def _fmt(x):
        s = f"{x:.1f}"
        return s[:-2] if s.endswith(".0") else s
    err_series = shares.apply(lambda r: f"{_fmt(r['_ins'])}/{_fmt(r['_del'])}/{_fmt(r['_sub'])}", axis=1)
    err_series.name = "Avg. Error (I/D/S %)"

    table = acc.join(err_series).rename(columns={"easy":"Easy","medium":"Medium","hard":"Hard"})
    num_cols = ["Easy","Medium","Hard","Average"]
    table[num_cols] = table[num_cols].astype(float).round(decimals)

    # preferred model row order
    order = ["flash","pro","gemini-2.5-flash","gemini-2.5-pro"]
    table = table.reindex([m for m in order if m in table.index] + [i for i in table.index if i not in order])
    return table

def generate_report(results_dir, out_dir="reports", decimals=2):
    os.makedirs(out_dir, exist_ok=True)
    df = load_results_dir(results_dir)

    # normalize fields
    df["DIFFICULTY"] = df["DIFFICULTY"].str.lower()
    df["ACC"] = pd.to_numeric(df["ACC"], errors="coerce")

    # build one big table across all topics
    rows = []
    for topic, df_t in df.groupby("TOPIC", sort=True):
        tname = pretty_topic_map[topic]
        ttable = build_topic_table(df_t, decimals=decimals)
        ttable.insert(0, "Topic", tname)
        rows.append(ttable.reset_index().rename(columns={"MODEL": "Model"}))

    big_table = pd.concat(rows, ignore_index=True)

    # sort by Topic, then model order
    model_order = {"flash": 0, "pro": 1, "gemini-2.5-flash": 0, "gemini-2.5-pro": 1}
    big_table["__ord"] = big_table["Model"].map(model_order).fillna(99)
    big_table = big_table.sort_values(by=["Topic","__ord","Model"]).drop(columns="__ord")

    # print to terminal
    print("\n=== MAPTIC Evaluation Table (merged) ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 140):
        print(big_table.to_string(index=False))

    # write single CSV
    out_csv = os.path.join(out_dir, "maptic_table_all_topics.csv")
    big_table.to_csv(out_csv, index=False)
    print(f"\nWrote CSV: {out_csv}")
    return big_table

# ---- CLI ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge result JSONs in a directory and produce a single CSV report.")
    parser.add_argument("results_dir", help="Directory containing JSON array result files.")
    parser.add_argument("--out_dir", default="reports", help="Output directory (default: reports)")
    parser.add_argument("--decimals", type=int, default=2, help="Accuracy rounding (default: 2)")
    args = parser.parse_args()
    generate_report(args.results_dir, args.out_dir, decimals=args.decimals)
