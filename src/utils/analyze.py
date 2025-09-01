# ---- CLI ----
"""
Usage:
    python src/utils/analyze.py path/to/results_dir --out_dir path/to/output_dir
Notes:
    - Reads every *.json in results_dir (top-level JSON arrays).
    - Computes *error rates* (100 - accuracy).
    - Terminal output: bolds the max difficulty error per row; underlines the min 'Average' within each topic.
    - Excel output with the same styling.
"""
import os, json, glob
import pandas as pd

REQUIRED_COLS = {"TOPIC","MODEL","DIFFICULTY","ACC","ERRORS"}

ANSI_BOLD = "\033[1m"
ANSI_UL   = "\033[4m"
ANSI_RESET= "\033[0m"

def load_results_dir(results_dir):
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

def pretty_topic(t):
    mapping = {
        "lower to upper string": "Lowercase→Uppercase (random)",
        "lower to upper words": "Lowercase→Uppercase (words)",
        "lower to upper natural text": "Lowercase→Uppercase (text)",
        "upper to lower string": "Uppercase→Lowercase (random)",
        "upper to lower words": "Uppercase→Lowercase (words)",
        "upper to lower natural text": "Uppercase→Lowercase (text)",
        "RNA": "RNA→Protein",
        "digits": "Digits→Number Words",
        "numbers": "Number Words→Digits",
        "country code to country": "ISO-3166 Code→Country",
    }
    return mapping.get(t, t)

def _format_shares_to_ids_percent(s_ins, s_del, s_sub):
    total = s_ins + s_del + s_sub
    if total == 0:
        return "–/–/–"
    shares = [(100*s_ins/total), (100*s_del/total), (100*s_sub/total)]
    def fmt(x):
        s = f"{x:.1f}"
        return s[:-2] if s.endswith(".0") else s
    return f"{fmt(shares[0])}/{fmt(shares[1])}/{fmt(shares[2])}"

def build_topic_table(df_topic, decimals=4, zero_tol=1e-12):
    # mean *error* per difficulty per model
    err = 100 - (
        df_topic.groupby(["MODEL","DIFFICULTY"])["ACC"]
        .mean()
        .unstack("DIFFICULTY")
        .reindex(columns=["easy","medium","hard"])
    )
    err = err.rename(columns={"easy":"Easy Err","medium":"Medium Err","hard":"Hard Err"})
    err["Average"] = err.mean(axis=1)

    # ---- NEW: Exact Match (%) across all examples in this topic (per model)
    df_em = df_topic.assign(
        _ins=df_topic["ERRORS"].apply(lambda e: e.get("INSERTIONS", 0)),
        _del=df_topic["ERRORS"].apply(lambda e: e.get("DELETIONS", 0)),
        _sub=df_topic["ERRORS"].apply(lambda e: e.get("SUBSTITUTIONS", 0)),
    )
    df_em["perfect"] = (df_em["_ins"] + df_em["_del"] + df_em["_sub"] == 0).astype(int)
    em = (df_em.groupby("MODEL")["perfect"].mean() * 100.0).rename("Exact Match (%)")

    # error composition (I/D/S)
    tmp = df_em.groupby("MODEL")[["_ins","_del","_sub"]].sum()
    ids_str = tmp.apply(lambda r: _format_shares_to_ids_percent(r["_ins"], r["_del"], r["_sub"]), axis=1)
    ids_str.name = "Error Breakdown (I/D/S %)"

    # combine
    out = err.join([em, ids_str])

    # display table: 0 → "–" for difficulty cells; keep numeric for Average and EM
    disp = out.copy()
    for col in ["Easy Err","Medium Err","Hard Err"]:
        disp[col] = disp[col].apply(
            lambda x: "–" if pd.isna(x) or abs(float(x)) <= zero_tol else f"{float(x):.{decimals}f}"
        )
    # numeric rounding for Average and Exact Match
    disp["Average"] = out["Average"].round(decimals)
    disp["Exact Match (%)"] = out["Exact Match (%)"].round(1)

    return out, disp

def apply_terminal_styling(df_disp, df_num, topic_name):
    """
    Bold the max of [Easy Err, Medium Err, Hard Err] per row.
    Underline the min 'Average' within this topic across rows (e.g., flash vs pro).
    """
    rows = []
    # find min Average inside this topic (numeric, ignoring NaN)
    avg_col = df_num["Average"]
    if len(avg_col.dropna()) > 0:
        min_avg = avg_col.min()
    else:
        min_avg = None

    for idx, row in df_disp.iterrows():
        # bold the largest difficulty error (numeric compare from df_num)
        e_vals = df_num.loc[idx, ["Easy Err","Medium Err","Hard Err"]]
        # index of max (if all zeros/NaNs, leave as-is)
        max_col = None
        if e_vals.notna().any():
            # Safely convert to numeric, coerce invalid to NaN, then fill NaN with -inf so a real number wins
            e_vals_num = pd.to_numeric(e_vals, errors="coerce").fillna(float("-inf"))
            max_col = e_vals_num.idxmax()

        styled = row.copy()
        for c in ["Easy Err","Medium Err","Hard Err"]:
            if row[c] != "–" and max_col == c:
                styled[c] = f"{ANSI_BOLD}{row[c]}{ANSI_RESET}"

        # underline the min Average in this topic
        if min_avg is not None and pd.notna(df_num.loc[idx, "Average"]) and df_num.loc[idx, "Average"] == min_avg:
            styled["Average"] = f"{ANSI_UL}{row['Average']}{ANSI_RESET}"

        rows.append(styled)
    return pd.DataFrame(rows, index=df_disp.index)

def write_excel_with_styles(big_disp, big_num, path_xlsx, decimals=4):
    """
    Single sheet 'MAPTIC':
      - Difficulty cells written as numbers when present, "–" as text.
      - 'Average' written as a number (allows sorting/filtering in Excel).
      - Bold the per-row max difficulty error.
      - Underline the per-topic min 'Average'.
    """
    import math
    with pd.ExcelWriter(path_xlsx, engine="xlsxwriter") as xw:
        wb = xw.book
        ws = wb.add_worksheet("MAPTIC")

        cols = list(big_disp.columns)
        col_idx = {c: i for i, c in enumerate(cols)}

        # Formats
        header_fmt     = wb.add_format({"bold": True})
        num_fmt_str    = "0." + "0"*decimals if decimals > 0 else "0"
        fmt_num        = wb.add_format({"num_format": num_fmt_str})
        fmt_num_bold   = wb.add_format({"num_format": num_fmt_str, "bold": True})
        fmt_num_ul     = wb.add_format({"num_format": num_fmt_str, "underline": 1})
        fmt_text       = wb.add_format()                 # plain text
        fmt_text_bold  = wb.add_format({"bold": True})   # not really used; here for completeness

        # Write header row
        for c, name in enumerate(cols):
            ws.write(0, c, name, header_fmt)

        # Precompute: per-topic min Average (numeric)
        # (robust: coerce to numeric)
        min_avg_by_topic = (
            big_num.assign(_avg=pd.to_numeric(big_num["Average"], errors="coerce"))
                   .groupby("Topic")["_avg"].min()
                   .to_dict()
        )

        # Write data rows with per-cell typing + styling
        for r, idx in enumerate(big_disp.index, start=1):
            topic = big_disp.at[idx, "Topic"]
            model = big_disp.at[idx, "Model"]

            # Determine per-row max difficulty (numeric)
            e_vals = pd.to_numeric(
                big_num.loc[idx, ["Easy Err", "Medium Err", "Hard Err"]],
                errors="coerce"
            )
            # idxmax with all-NaNs would raise; guard it
            max_col = None
            if e_vals.notna().any():
                max_col = e_vals.idxmax()

            # Underline condition for Average
            avg_num = pd.to_numeric(big_num.at[idx, "Average"], errors="coerce")
            underline_avg = (pd.notna(avg_num) and
                             topic in min_avg_by_topic and
                             math.isclose(avg_num, min_avg_by_topic[topic], rel_tol=0, abs_tol=1e-12))

            for c, col in enumerate(cols):
                val_disp = big_disp.at[idx, col]

                if col in ("Topic", "Model", "Error Breakdown (I/D/S %)"):
                    # Always text
                    ws.write_string(r, c, str(val_disp), fmt_text)
                    continue

                if col in ("Easy Err", "Medium Err", "Hard Err"):
                    # "–" → text; else numeric with optional bold
                    if isinstance(val_disp, str) and val_disp.strip() == "–":
                        ws.write_string(r, c, "–", fmt_text)
                    else:
                        num = pd.to_numeric(big_num.at[idx, col], errors="coerce")
                        if pd.isna(num):
                            ws.write_string(r, c, "–", fmt_text)
                        else:
                            use_fmt = fmt_num_bold if (max_col == col) else fmt_num
                            ws.write_number(r, c, float(num), use_fmt)
                    continue

                if col == "Average":
                    # Always numeric; underline if this is the topic's min
                    if pd.isna(avg_num):
                        ws.write_string(r, c, "–", fmt_text)
                    else:
                        use_fmt = fmt_num_ul if underline_avg else fmt_num
                        ws.write_number(r, c, float(avg_num), use_fmt)
                    continue
            
                if col == "Exact Match (%)":
                    em_num = pd.to_numeric(big_num.at[idx, col], errors="coerce")
                    if pd.isna(em_num):
                        ws.write_string(r, c, "–", fmt_text)
                    else:
                        ws.write_number(r, c, float(em_num), fmt_num)
                    continue

                # Fallback (shouldn't happen): write as text
                ws.write_string(r, c, str(val_disp), fmt_text)

        # Optional: column widths
        for c, name in enumerate(cols):
            width = max(len(name), 14)
            if name in ("Topic", "Model", "Error Breakdown (I/D/S %)"):
                width = max(width, 22)
            ws.set_column(c, c, width)

def generate_report(results_dir, out_dir="reports", decimals=4, zero_tol=1e-12):
    os.makedirs(out_dir, exist_ok=True)
    df = load_results_dir(results_dir)

    # normalize fields
    df["DIFFICULTY"] = df["DIFFICULTY"].str.lower()
    df["ACC"] = pd.to_numeric(df["ACC"], errors="coerce")

    # build big table
    blocks_disp = []
    blocks_num  = []
    for topic, df_t in df.groupby("TOPIC", sort=True):  # keep natural topic order
        tname = pretty_topic(topic)
        num, disp = build_topic_table(df_t, decimals=decimals, zero_tol=zero_tol)
        # Add Topic/Model columns
        num = num.reset_index().rename(columns={"MODEL":"Model"})
        disp = disp.reset_index().rename(columns={"MODEL":"Model"})
        num.insert(0, "Topic", tname)
        disp.insert(0, "Topic", tname)
        blocks_num.append(num)
        blocks_disp.append(disp)

    big_num  = pd.concat(blocks_num,  ignore_index=True)
    big_disp = pd.concat(blocks_disp, ignore_index=True)

    # order models flash→pro if present (no sorting of topics)
    model_order = {"flash": 0, "pro": 1, "gemini-2.5-flash": 0, "gemini-2.5-pro": 1}
    big_num["__ord"]  = big_num["Model"].map(model_order).fillna(99)
    big_disp["__ord"] = big_disp["Model"].map(model_order).fillna(99)

    # stable sort by appearance of Topic only; keep model order per topic
    # (no sorting by Topic; we only order within-topic rows)
    big_num  = big_num.sort_values(by=["Topic","__ord","Model"]).drop(columns="__ord")
    big_disp = big_disp.sort_values(by=["Topic","__ord","Model"]).drop(columns="__ord")

    # terminal styling
    term_disp = big_disp.copy()
    term_disp = apply_terminal_styling(term_disp, big_num, topic_name=None)

    print("\n=== MAPTIC Evaluation Table (Error Rates; bold=max per row, underline=min Avg per topic) ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(term_disp.to_string(index=False))

    # write CSV (un-styled, machine-friendly)
    csv_path = os.path.join(out_dir, "maptic_table_error_rates.csv")
    big_disp.to_csv(csv_path, index=False)
    print(f"\nWrote CSV: {csv_path}")

    # write Excel with styling
    xlsx_path = os.path.join(out_dir, "maptic_table_error_rates.xlsx")
    write_excel_with_styles(big_disp, big_num, xlsx_path, decimals=decimals)
    print(f"Wrote Excel: {xlsx_path}")

    return big_disp

# ---- CLI ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge result JSONs and output error-rate table with styling.")
    parser.add_argument("results_dir", help="Directory containing JSON array result files.")
    parser.add_argument("--out_dir", default="reports", help="Output directory (default: reports)")
    parser.add_argument("--decimals", type=int, default=4, help="Decimal places for error rates (default: 4)")
    parser.add_argument("--zero_tol", type=float, default=1e-12, help="Treat |error| ≤ tol as zero (default: 1e-12)")
    args = parser.parse_args()
    generate_report(args.results_dir, args.out_dir, decimals=args.decimals, zero_tol=args.zero_tol)
