#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import pandas as pd

EXPECTED_COLS = ["url", "comment", "language", "label"]

def sniff_delimiter(path, sample_bytes=65536):
    with open(path, "rb") as f:
        sample = f.read(sample_bytes).decode("utf-8", "ignore")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        return ","

def main():
    #ap = argparse.ArgumentParser(description="Convert CSV (url,comment,language,label) to Excel.")
    #ap.add_argument("csv_in", help="Input CSV file")
    #ap.add_argument("-o", "--xlsx_out", help="Output .xlsx path (default: same name as input)")
    #ap.add_argument("--sheet", default="SATD", help="Excel sheet name (default: SATD)")
    # ap.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default: utf-8-sig)")
    #args = ap.parse_args()

    csv_path = ".//SATD_comments.csv"
    
    out_path = "./forFixed/SATD_brut_Excel.xlsx"
    delim = sniff_delimiter(csv_path)

    # Read, preserving text exactly (no NA coercion), keep everything as strings
    df = pd.read_csv(
        csv_path,
        delimiter=delim,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    # Ensure expected columns (reorder, add missing if necessary)
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[EXPECTED_COLS]

    # Write Excel with formatting
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="SATD")
        ws = writer.sheets["SATD"]
        wb = writer.book

        # Formats
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        header = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
        url_fmt = wb.add_format({"font_color": "blue", "underline": 1})

        # Header format
        for col_idx, _ in enumerate(EXPECTED_COLS):
            ws.write(0, col_idx, EXPECTED_COLS[col_idx], header)

        # Column widths & wrapping
        col_widths = {
            "url": 60,
            "comment": 80,
            "language": 16,
            "label": 14,
        }
        for idx, name in enumerate(EXPECTED_COLS):
            if name == "comment":
                ws.set_column(idx, idx, col_widths[name], wrap)
            else:
                ws.set_column(idx, idx, col_widths[name])

        # Freeze header row
        ws.freeze_panes(1, 0)

        # Make the URL column clickable
        url_col_idx = EXPECTED_COLS.index("url")
        for r, link in enumerate(df["url"], start=1):
            if isinstance(link, str) and link.strip().lower().startswith(("http://", "https://")):
                ws.write_url(r, url_col_idx, link, url_fmt, string=link)

    print(f"✅ Wrote Excel → {out_path}")

if __name__ == "__main__":
    main()
