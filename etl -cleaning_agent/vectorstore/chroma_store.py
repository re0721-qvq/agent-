# vectorstore/chroma_store.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config.settings import CHROMA_DIR, DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, EMBEDDING_MODEL

# 初始化百炼的 Embedding 模型
_emb = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    check_embedding_ctx_length=False,
)


def get_store(collection: str) -> Chroma:
    """获取指定集合的 Chroma 向量库实例"""
    return Chroma(
        collection_name=collection,
        embedding_function=_emb,
        persist_directory=str(CHROMA_DIR),
    )


def remember_schema(table_name: str, schema_desc: str) -> None:
    """将表结构存入向量库（长期记忆）"""
    store = get_store("schemas")
    store.add_documents([
        Document(page_content=schema_desc, metadata={"table_name": table_name})
    ])


def recall_schema(query: str, k: int = 2) -> str:
    """根据自然语言查询，从向量库中召回最相关的历史表结构"""
    store = get_store("schemas")
    try:
        docs = store.similarity_search(query, k=k)
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        # 如果向量库为空或者发生异常，返回空字符串，不影响主流程
        print(f"[Chroma] recall_schema 失败: {e}")
        return ""


def remember_rule(rule: str, metadata: dict | None = None) -> None:
    """将历史清洗规则存入向量库（长期记忆）"""
    store = get_store("rules")
    store.add_documents([Document(page_content=rule, metadata=metadata or {})])


def recall_rules(query: str, k: int = 2) -> str:
    """根据自然语言查询，召回相关的历史清洗规则"""
    store = get_store("rules")
    try:
        docs = store.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in docs)
    except Exception as e:
        print(f"[Chroma] recall_rules 失败: {e}")
        return ""