#core/state.py
from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ETLState(TypedDict):
    messages: Annotated[list, add_messages]
    df_path: Optional[str]
    table_name: Optional[str]
    schema_desc: Optional[str]
    last_result_path: Optional[str]


class SQLInput(BaseModel):
    question: str = Field(..., description="自然语言查询或聚合需求")
    table_name: str = Field(..., description="DuckDB 视图名")


class QualityInput(BaseModel):
    table_name: str = Field(..., description="要校验的表名")


class PandasCleanInput(BaseModel):
    instruction: str = Field(..., description="自然语言清洗指令")
    table_name: str = Field(..., description="要清洗的表名")


class VizInput(BaseModel):
    question: str = Field(..., description="图表标题或指标")
    table_name: str = Field(..., description="数据表名")
    kind: str = Field("bar", description="bar / line / pie / scatter")