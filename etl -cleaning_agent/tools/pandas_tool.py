# tools/pandas_tool.py
import pandas as pd
from typing import Annotated, Optional
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState

from config.settings import DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, QWEN_MODEL, PARQUET_DIR
from core.state import PandasCleanInput
from tools.quality_tool import _load

llm = ChatOpenAI(
    model=QWEN_MODEL, base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY, temperature=0,
    timeout=120,        # 防无限等待
    max_retries=1,
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是 Pandas 清洗专家。输入 DataFrame 变量名 df，执行后必须重新赋值给 df。"
     "只输出代码，不要 markdown 代码块，不要 import（pd 已预置）。"),
    ("human", "列: {columns}\n前 5 行:\n{head}\n\n指令: {instruction}\n\n代码:"),
])


@tool(args_schema=PandasCleanInput)
def pandas_clean(instruction: str, table_name: str,
                 state: Annotated[Optional[dict], InjectedState] = None) -> str:
    """按自然语言指令执行 Pandas 清洗（缺失值、类型转换、去重、列重命名等），
    清洗后的数据会写回 parquet 文件。适用于行级、结构级的 DataFrame 修改。"""
    print(f"\n[pandas_clean] 开始，table={table_name}")

    st = state or {}                       # ← 关键：先把 None 兜底成 dict
    df = _load(st, table_name)
    if df is None:
        print("[pandas_clean] 加载失败")
        return "无法加载数据表。"
    print(f"[pandas_clean] 加载完成，{len(df)} 行 × {len(df.columns)} 列")

    print("[pandas_clean] 调用 LLM 生成代码...")
    raw = (_PROMPT | llm).invoke({
        "columns": list(df.columns),
        "head": df.head().to_string(),
        "instruction": instruction,
    }).content.strip()
    print(f"[pandas_clean] LLM 返回，{len(raw)} 字符")

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    local = {"df": df.copy(), "pd": pd}
    try:
        print(f"[pandas_clean] 执行代码，{len(raw.splitlines())} 行")
        exec(raw, {"pd": pd}, local)
        print("[pandas_clean] 执行完成")
        cleaned = local["df"]
        if not isinstance(cleaned, pd.DataFrame):
            return "清洗未返回 DataFrame。"
    except Exception as e:
        print(f"[pandas_clean] 执行失败: {e}")
        return f"执行失败：{e}\n代码：\n{raw}"

    out = PARQUET_DIR / f"{table_name}_cleaned.parquet"
    cleaned.to_parquet(out, index=False)
    print(f"[pandas_clean] 写回完成: {out}")
    return (f"清洗完成：{len(df)} → {len(cleaned)} 行，"
            f"{len(df.columns)} → {len(cleaned.columns)} 列。\n已保存: {out}")