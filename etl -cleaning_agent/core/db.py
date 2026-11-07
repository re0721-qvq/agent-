# core/db.py
import duckdb
from config.settings import DUCKDB_PATH


def get_conn() -> duckdb.DuckDBPyConnection:
    """所有工具统一用这个连接 DuckDB，保证视图共享。"""
    return duckdb.connect(str(DUCKDB_PATH))