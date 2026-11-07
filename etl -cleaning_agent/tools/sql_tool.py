# tools/sql_tool.py
import re
from pathlib import Path
from typing import Annotated, Optional

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState

from config.settings import DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, QWEN_MODEL, PARQUET_DIR
from core.state import SQLInput
from core.db import get_conn
# 导入 Chroma 的召回方法
from vectorstore.chroma_store import recall_schema

llm = ChatOpenAI(
    model=QWEN_MODEL, base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY, temperature=0,
    timeout=240,
    max_retries=0,
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是 DuckDB SQL 专家。只输出一条可执行 SELECT 语句。\n"
     "硬性规则：\n"
     "1. 表名和所有列名必须用双引号包裹。\n"
     "2. 不要编造列名，只用给定的列。\n"
     "3. 字符串用单引号，数值/日期必要时用 CAST 或 TRY_CAST。\n"
     "4. 日期格式化用 strftime，聚合连接用 string_agg。\n"
     "5. 只输出 SQL 本身，不要 markdown、不要注释、不要解释。\n"
     "其余维度在 SQL 里用注释说明'需要单独查询'，不要用 UNION ALL 硬拼。\n"
     "6. 如果问题涉及多个独立统计维度，优先选择其中最主要的一个其余维度在 SQL 里用注释说明'需要单独查询'，不要用 UNION ALL 硬拼。\n"
     "7. SQL 长度控制在 500 字符以内，超长说明问题该拆分。\n"
     "8. 必须严格区分指标：用户问'销售额/销量/Sales'时，必须用 Sales 列；用户问'利润/利润率/Profit'时，必须用 Profit 列。绝对不能混淆！\n"),
    ("human",
     "表名: {table_name}\n\n可用列:\n{columns}\n\n表结构样本:\n{schema_desc}\n\n问题: {question}\n\nSQL:"),
])


def _clean_sql(raw: str) -> str:
    """去掉 markdown fence、行首注释、多余分号，只保留第一条 SELECT。"""
    s = raw.strip()
    # 剥 markdown fence
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    # 删行首 -- 注释
    s = "\n".join(line for line in s.splitlines() if not line.strip().startswith("--"))
    s = s.strip().rstrip(";").strip()
    # 若有多条语句，取第一条 SELECT
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        selects = [p for p in parts if p.upper().startswith("SELECT")]
        if selects:
            s = selects[0]
    return s


@tool(args_schema=SQLInput)
def sql_query(question: str, table_name: str,
              state: Annotated[Optional[dict], InjectedState] = None) -> str:
    """生成 DuckDB SQL 并执行，结果保存为 parquet，返回预览。"""
    st = state or {}

    # 1. 获取当前状态中的 schema，并尝试 RAG 召回
    current_schema = st.get("schema_desc", "")

    try:
        # 从向量库中检索与当前问题最相关的历史表结构（召回前 2 个）
        recalled_schema = recall_schema(question, k=2)
    except Exception as e:
        print(f"[sql_query] RAG 召回失败: {e}")
        recalled_schema = ""

    # 2. 合并 Schema（优先保留当前表，附加历史相关表）
    if current_schema:
        if recalled_schema and recalled_schema not in current_schema:
            schema = f"【当前表结构】\n{current_schema}\n\n【历史参考表结构】\n{recalled_schema}"
        else:
            schema = current_schema
    else:
        schema = recalled_schema

    if not schema:
        return "缺少表结构描述，且未能在向量库中召回相关表结构。现有表: 未知"

    # 3. 提取列名（仅从当前表的 schema 中提取，防止 RAG 引入无关列名）
    cols = []
    if current_schema:
        cols = re.findall(r"^\s*-\s+(.+?)\s+\(", current_schema, re.MULTILINE)
    else:
        # 如果没有当前表 schema，只能从所有召回中尽力提取
        cols = re.findall(r"^\s*-\s+(.+?)\s+\(", schema, re.MULTILINE)

    columns_text = "\n".join(f'  - "{c.strip()}"' for c in cols) if cols else "（未解析）"

    # 4. 调用 LLM 生成 SQL
    raw = (_PROMPT | llm).invoke({
        "question": question,
        "table_name": table_name,
        "columns": columns_text,
        "schema_desc": schema,  # 喂入合并后的 schema
    }).content.strip()
    sql = _clean_sql(raw)

    print(f"\n{'=' * 70}")
    print(f"[sql_query] table={table_name}")
    print(f"[sql_query] 列名: {cols}")
    print(f"[sql_query] RAG 召回: {'已找到' if recalled_schema else '无'}")
    print(f"[sql_query] SQL:\n{sql}")

    con = get_conn()
    try:
        existing = con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).df()["table_name"].tolist()

        # 兜底：视图不存在时从 df_path 重建
        if table_name not in existing:
            df_path = st.get("df_path")
            if isinstance(df_path, str) and Path(df_path).exists():
                safe_path = df_path.replace("\\", "/").replace("'", "''")
                con.execute(
                    f'CREATE OR REPLACE VIEW "{table_name}" '
                    f"AS SELECT * FROM read_parquet('{safe_path}')"
                )
                print(f"[sql_query] 视图 {table_name} 不存在，已从 {df_path} 重建")
                existing = con.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).df()["table_name"].tolist()
            if table_name not in existing:
                print(f"[sql_query] 表仍不存在。现有表: {existing}")
                return f"表 `{table_name}` 不存在，且 df_path 无效。现有表: {existing}"

        df = con.execute(sql).df()
        print(f"[sql_query] 成功，{len(df)} 行 × {len(df.columns)} 列")
    except Exception as e:
        err = str(e)
        print(f"[sql_query] 失败: {type(e).__name__}: {err}")
        try:
            real_cols = con.execute(f'DESCRIBE "{table_name}"').df()["column_name"].tolist()
            err += f"\n该表实际列名: {real_cols}"
        except Exception:
            pass
        return (
            f"SQL 执行失败：{err}\n\n"
            f"生成的 SQL:\n{sql}\n\n"
            f"请根据错误修正后重试，不要重复生成相同的 SQL。"
        )
    finally:
        con.close()
    print(f"{'=' * 70}")

    out = PARQUET_DIR / "last_sql_result.parquet"
    df.to_parquet(out, index=False)
    return (f"SQL:\n{sql}\n\n结果（前 {min(20, len(df))} 行 / 共 {len(df)} 行）:\n"
            f"{df.head(20).to_string(index=False)}\n\n已保存: {out}")