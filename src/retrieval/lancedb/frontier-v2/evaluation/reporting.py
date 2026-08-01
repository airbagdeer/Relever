"""Markdown formatting helpers for the evaluation reports.

Matches the house style of ``summary/minimax/SUMMARY.md``: a metadata header,
markdown tables with the top-3 values per column flagged with 🟩, ``—`` for
missing cells, and ``***``/``**``/``*``/``ns`` significance markers.
"""

from __future__ import annotations

import pandas as pd

TOP_MARK = "🟩"


def fmt_num(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    # Render integer-valued numbers as ints (pandas .iterrows() upcasts int columns
    # to float, so counts like n_queries arrive here as 120.0 — show them as "120").
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        xf = float(x)
        return str(int(xf)) if xf.is_integer() else f"{xf:.{nd}f}"
    return str(x)


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 1e-4:
        return "<1e-4"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def markdown_table(
    df: pd.DataFrame,
    highlight_cols: list[str] | None = None,
    nd: int = 4,
    index_name: str | None = None,
    top_n: int = 3,
) -> str:
    """Render a DataFrame as a GitHub markdown table, flagging the top-`top_n`
    values in each `highlight_cols` column with 🟩."""
    highlight_cols = highlight_cols or []
    df = df.copy()

    top_sets: dict[str, set] = {}
    for c in highlight_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            top_sets[c] = set(df[c].dropna().nlargest(top_n).tolist())

    header_cols = ([index_name] if index_name is not None else []) + list(df.columns)
    lines = ["| " + " | ".join(header_cols) + " |",
             "|" + "|".join(["---"] * len(header_cols)) + "|"]
    for idx, row in df.iterrows():
        cells = []
        if index_name is not None:
            cells.append(str(idx))
        for c in df.columns:
            val = row[c]
            txt = fmt_num(val, nd)
            if c in top_sets and isinstance(val, (int, float)) and not pd.isna(val) and val in top_sets[c]:
                txt = f"{TOP_MARK} {txt}"
            cells.append(txt)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def metadata_header(title: str, lines: list[str]) -> str:
    body = "\n".join(f"- {ln}" for ln in lines)
    return f"# {title}\n\n{body}\n"


def corr_strength(rho: float) -> str:
    a = abs(rho)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    if a < 0.7:
        return "strong"
    return "very strong"
