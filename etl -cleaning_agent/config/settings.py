#config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"
FIG_DIR = DATA_DIR / "figures"
CHROMA_DIR = DATA_DIR / "chroma"
for d in (PARQUET_DIR, FIG_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = "qwen3.8-max"
EMBEDDING_MODEL = "text-embedding-v3"
DUCKDB_PATH = DATA_DIR / "etl.duckdb"