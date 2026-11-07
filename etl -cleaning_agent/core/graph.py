# core/graph.py
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver  # 引入内存保存器

from core.state import ETLState
from config.settings import DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, QWEN_MODEL
from tools.sql_tool import sql_query
from tools.quality_tool import check_quality
from tools.pandas_tool import pandas_clean
from tools.viz_tool import visualize

TOOLS = [sql_query, check_quality, pandas_clean, visualize]
llm = ChatOpenAI(
    model=QWEN_MODEL, base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY, temperature=0,
).bind_tools(TOOLS)

SYSTEM = """你是 ETL 智能清洗 Agent。

当前表: {table_name}
结构:
{schema_desc}

工具：sql_query / check_quality / pandas_clean / visualize。
规则：
1.数值聚合用 sql_query；结构清洗用 pandas_clean；
2.无明确指令先 check_quality；一次只调一个工具，简洁中文总结，不编造数据。
3.遇到“汇总、分组、排序、Top N”等分析需求（例如“按xx汇总销售额，取前20”），**必须**先调用 sql_query 生成聚合后的数据表。
4.只有在数据已经汇总好，或者用户要求直接对原始数据画分布图时，才调用 visualize。
5.一次 sql_query 只回答一个问题。如果用户的问题包含"1) ... 2) ... 3) ..."或多个独立统计需求，先拆成多次 sql_query 依次执行，不要试图用一条 SQL 完成。
6.sql_query 生成的是单条 SELECT，禁止 UNION ALL 拼接多个不相关查询。
7.用户要求"明细样例"和"聚合统计"同时出现时，先做聚合，再做明细。
"""


def agent_node(state: ETLState):
    sys = SystemMessage(content=SYSTEM.format(
        table_name=state.get("table_name", "未命名"),
        schema_desc=state.get("schema_desc", "暂无"),
    ))
    return {"messages": [llm.invoke([sys] + state["messages"])]}


def build_graph():
    g = StateGraph(ETLState)  # type: ignore[arg-type]
    g.add_node("agent", agent_node)  # type: ignore[arg-type]
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    # 实例化内存保存器，并在编译时传入
    memory = MemorySaver()
    return g.compile(checkpointer=memory)


ETL_GRAPH = build_graph()


def run_etl(user_message: str, table_name: str, df_path: str, schema_desc: str, thread_id: str) -> dict:
    """
    增加 thread_id 参数。属于同一个 thread_id 的对话会共享短期记忆。
    """
    init = {
        "messages": [("user", user_message)],
        "table_name": table_name,
        "df_path": df_path,
        "schema_desc": schema_desc,
    }

    # 构造 config，指定 thread_id 和递归限制
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50
    }

    # 执行图流
    for step, chunk in enumerate(ETL_GRAPH.stream(init, config=config)):  # type: ignore[arg-type]
        for node_name, node_output in chunk.items():
            print(f"\n--- [步骤 {step + 1}] 节点: {node_name} ---")

            if not isinstance(node_output, dict):
                continue

            msgs = node_output.get("messages") or []
            for msg in msgs:
                msg_type = type(msg).__name__
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", None) or []

                if tool_calls:
                    for tc in tool_calls:
                        print(f"  [{msg_type}] 调用工具: {tc['name']}")
                        print(f"    参数: {tc['args']}")
                elif content:
                    preview = content[:300].replace("\n", " ")
                    print(f"  [{msg_type}] {preview}")

    final_state = ETL_GRAPH.get_state(config).values

    return dict(final_state)