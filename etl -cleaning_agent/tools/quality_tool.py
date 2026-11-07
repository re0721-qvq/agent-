# tools/quality_tool.py
import pandas as pd
from pathlib import Path
from typing import Annotated, Optional, cast

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.state import QualityInput
from core.db import get_conn


def _load(state: dict, table_name: str) -> Optional[pd.DataFrame]:
    p = state.get("df_path")
    if isinstance(p, str) and Path(p).exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    try:
        con = get_conn()
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
        con.close()
        return df
    except Exception:
        return None


@tool(args_schema=QualityInput)
def check_quality(
    table_name: str,
    state: Annotated[Optional[dict], InjectedState] = None,
) -> str:
    """数据质量校验：行数、缺失率、重复行、数值列 3σ 异常值。"""
    st = state or {}
    df = _load(st, table_name)
    if df is None:
        return "无法加载数据表。"

    lines = [f"表 `{table_name}` 质量报告",
             f"行数: {len(df)}", f"列数: {len(df.columns)}", "",
             "【缺失率 Top 10】"]
    for col, r in df.isna().mean().sort_values(ascending=False).head(10).items():
        if r > 0:
            lines.append(f"  {col}: {r:.1%}")

    lines.append(f"\n【重复行】: {df.duplicated().sum()} 行")
    lines.append("\n【数值列 3σ 异常值】")
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        lines.append("  无数值列")
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 10 or s.std() == 0:
            continue
        n = int(cast(float, ((s - s.mean()).abs() > 3 * s.std()).sum()))
        if n > 0:
            lines.append(f"  {col}: {n} 个")
    return "\n".join(lines)