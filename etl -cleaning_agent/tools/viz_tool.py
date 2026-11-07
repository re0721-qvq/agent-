# tools/viz_tool.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated, Optional, Any
import re

from config.settings import FIG_DIR, PARQUET_DIR  # 引入 PARQUET_DIR
from core.state import VizInput

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@tool(args_schema=VizInput)
def visualize(
        question: str,
        table_name: str,
        kind: str = "bar",
        state: Annotated[Optional[dict], InjectedState] = None,
) -> str:
    """根据最近一次结果生成 bar / line / pie / scatter 图表，自动处理数据截断防重叠。"""

    # 核心修复：优先从 SQL 结果文件中读取数据
    df = None
    sql_result_path = PARQUET_DIR / "last_sql_result.parquet"

    if sql_result_path.exists():
        try:
            df = pd.read_parquet(sql_result_path)
        except Exception:
            pass

    # 如果 SQL 结果不存在，再回退到图状态里的 df_path
    if df is None and state:
        p = state.get("df_path")
        if isinstance(p, str) and Path(p).exists():
            try:
                df = pd.read_parquet(p)
            except Exception:
                pass

    if df is None or df.empty:
        return "没有可用结果数据，请先执行 SQL 或清洗。"

    num_cols: list[Any] = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return "结果中没有数值列，无法绘图。"

    cat_cols: list[Any] = [c for c in df.columns if c not in num_cols]

    # 1. 智能匹配列名
    def find_col(keywords: list[str], cols: list[Any]) -> Optional[Any]:
        for kw in keywords:
            for c in cols:
                if kw in str(c).lower():
                    return c
        return None

    # 动态解析用户问题，决定 Y 轴指标的优先级
    q_lower = question.lower()
    sales_kws = ["sales", "amount", "销售额", "金额", "销量", "revenue", "total sales"]
    profit_kws = ["profit", "margin", "利润", "率", "total profit"]

    # 如果问的是销售额，优先找销售额；如果问的是利润，优先找利润
    if any(k in q_lower for k in ["销售额", "销量", "sales", "amount"]):
        y_candidates = sales_kws + profit_kws
    elif any(k in q_lower for k in ["利润", "profit", "margin"]):
        y_candidates = profit_kws + sales_kws
    else:
        # 默认优先销售额
        y_candidates = sales_kws + profit_kws

    x_col = find_col(["product", "name", "category", "产品", "名称", "类别"], cat_cols)
    y_col = find_col(y_candidates, num_cols)

    if x_col is None:
        x_col = cat_cols[0] if cat_cols else df.columns[0]
    if y_col is None:
        y_col = num_cols[0]

    col_x = str(x_col)
    col_y = str(y_col)

    # 调试打印：让你在 PyCharm 控制台能清楚看到工具选了什么列
    print(f"\n[visualize] 数据源列名: {df.columns.tolist()}")
    print(f"[visualize] 最终选中 -> 类别X: {col_x}, 数值Y: {col_y}")

    # 核心优化：判断数据源如果是已经聚合好的 SQL 结果，就不再强行 groupby
    # 只有读取的是包含大量重复行的原始数据（比如超过100行），才做兜底聚合
    is_raw_data = (not sql_result_path.exists()) or (len(df) > 1000)
    if is_raw_data and (len(df) > 100 or df[col_x].duplicated().sum() > 0):
        df = df.groupby(col_x, as_index=False)[col_y].sum()

    # 2. 提取 Top N
    max_bars = 20
    match = re.search(r'[Tt]op\s*(\d+)', question)
    if match:
        max_bars = int(match.group(1))

    df = df.sort_values(by=col_y, ascending=False).head(max_bars)
    question = f"{question} (前 {min(len(df), max_bars)} 条)"

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        if kind == "line":
            ax.plot(df[col_x].astype(str), df[col_y], marker="o")
            ax.tick_params(axis="x", rotation=45)

        elif kind == "pie":
            df_pie = df.head(10) if len(df) > 10 else df
            ax.pie(df_pie[col_y], labels=df_pie[col_x], autopct="%1.1f%%")

        elif kind == "scatter":
            ax.scatter(df[col_x], df[col_y])

        else:  # bar 柱状图
            is_long_labels = len(df) > 10 or any(len(str(v)) > 4 for v in df[col_x].head(3))

            if is_long_labels:
                ax.barh(df[col_x].astype(str)[::-1], df[col_y][::-1], color="steelblue")
                ax.set_xlabel(col_y)
                ax.set_ylabel(col_x)
            else:
                ax.bar(df[col_x].astype(str), df[col_y], color="steelblue")
                ax.tick_params(axis="x", rotation=45)

        ax.set_title(question)
        fig.tight_layout()
        out = FIG_DIR / "last_chart.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return f"图表已生成: {out}"
    except Exception as e:
        plt.close(fig)
        return f"绘图失败：{e}"