#ingest/web_scraper.py
import requests
import pandas as pd
from bs4 import BeautifulSoup
from config.settings import PARQUET_DIR
from vectorstore.chroma_store import remember_schema
from core.db import get_conn


def scrape_table(url: str, table_index: int = 0) -> pd.DataFrame:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    tables = BeautifulSoup(resp.text, "html.parser").find_all("table")
    if not tables:
        raise ValueError("页面中未找到 <table>")
    trs = tables[table_index].find_all("tr")
    if not trs:
        raise ValueError("目标表格为空")
    header = [th.get_text(strip=True) for th in trs[0].find_all(["th", "td"])]
    rows = []
    for tr in trs[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) == len(header):
            rows.append(cells)
    return pd.DataFrame(rows, columns=header)


def register_df(df: pd.DataFrame, table_name: str) -> tuple[str, str]:
    """parquet → DuckDB 视图 → schema 描述 → Chroma。"""
    path = PARQUET_DIR / f"{table_name}.parquet"
    df.to_parquet(path, index=False)

    # DuckDB 的 CREATE VIEW 不支持预编译参数，只能拼接
    # 对路径中的单引号做转义，防止 SQL 被破坏
    safe_path = path.as_posix().replace("'", "''")

    con = get_conn()
    try:
        con.execute(
            f'CREATE OR REPLACE VIEW "{table_name}" '
            f"AS SELECT * FROM read_parquet('{safe_path}')"
        )
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        print(f"[register_df] 视图 {table_name} 创建{'成功' if exists else '失败'}")
    finally:
        con.close()

    lines = [f"表名: {table_name}", f"行数: {len(df)}", "列:"]
    for c in df.columns:
        lines.append(f"  - {c} ({df[c].dtype})")
    schema = "\n".join(lines)
    remember_schema(table_name, schema)
    return str(path), schema