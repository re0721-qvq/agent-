# app.py
import uuid
import gradio as gr
import pandas as pd
from pathlib import Path

from config.settings import PARQUET_DIR, FIG_DIR
from core.graph import run_etl
from ingest.web_scraper import scrape_table, register_df
from typing import Optional, cast

CHART_PATH = FIG_DIR / "last_chart.png"


def handle_upload(file: Optional[str], table_name: str):
    if file is None:
        return "请上传文件。", "", None

    name = (table_name or "uploaded_data").strip()
    suffix = Path(file).suffix.lower()

    if suffix == ".csv":
        df = cast(pd.DataFrame, pd.read_csv(file))
    elif suffix in (".xlsx", ".xlsm"):
        df = cast(pd.DataFrame, pd.read_excel(file, engine="openpyxl"))
    elif suffix == ".xls":
        df = cast(pd.DataFrame, pd.read_excel(file, engine="calamine"))
    else:
        return "仅支持 CSV / Excel。", "", None

    _, schema = register_df(df, name)
    return f"已加载 `{name}`：{len(df)} 行。\n\n{schema}", schema, df.head(50)


def handle_scrape(url: str, table_index: int, table_name: str):
    if not url.strip():
        return "请输入 URL。", "", None
    name = (table_name or "web_data").strip()
    try:
        df = scrape_table(url.strip(), int(table_index))
        _, schema = register_df(df, name)
        return f"采集完成 `{name}`：{len(df)} 行。\n\n{schema}", schema, df.head(50)
    except Exception as e:
        return f"采集失败：{e}", "", None


# 函数签名增加 thread_id
def chat_fn(message, history, table_name, df_path, schema_desc, thread_id):
    if not message.strip():
        return history, "", gr.update()

    if not table_name or not schema_desc:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "请先加载数据。"},
        ], "", gr.update()

    # 记录调用前图表的修改时间
    before_mtime = CHART_PATH.stat().st_mtime if CHART_PATH.exists() else 0

    # 传入 thread_id
    result = run_etl(
        user_message=message,
        table_name=table_name,
        df_path=df_path,
        schema_desc=schema_desc,
        thread_id=thread_id
    )

    ai_msg = result["messages"][-1]
    answer = ai_msg.content or ""
    reasoning = ai_msg.additional_kwargs.get("reasoning_content", "")

    display = answer
    if reasoning:
        display = f"**思考过程**\n\n{reasoning}\n\n---\n\n{answer}"

    # 判断本次是否生成了新图
    after_mtime = CHART_PATH.stat().st_mtime if CHART_PATH.exists() else 0
    chart = str(CHART_PATH) if after_mtime > before_mtime else gr.update()

    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": display},
    ], "", chart


with gr.Blocks(title="ETL 智能清洗 Agent") as demo:
    gr.Markdown("## ETL 智能清洗 Agent\nqwen3.8-max · LangGraph · DuckDB · Chroma")

    tn = gr.State("")
    dp = gr.State("")
    sc = gr.State("")
    # 增加一个专门保存当前会话 ID 的 State
    tid = gr.State("")

    with gr.Tab("上传文件"):
        f_in = gr.File(
            label="CSV / Excel",
            file_types=[".csv", ".xlsx", ".xls"],
            file_count="single",
            type="filepath",
        )
        t_up = gr.Textbox(label="表名", value="uploaded_data",placeholder="请使用英文加下划线命名表的名字，不要用中文或空格")
        b_up = gr.Button("加载", variant="primary")
        info_up = gr.Textbox(label="信息", lines=6)
        prev_up = gr.Dataframe(label="预览", interactive=False)

    with gr.Tab("页面采集"):
        url = gr.Textbox(label="URL")
        idx = gr.Number(label="第几个表格", value=0, precision=0)
        t_sc = gr.Textbox(label="表名", value="web_data")
        b_sc = gr.Button("采集", variant="primary")
        info_sc = gr.Textbox(label="信息", lines=6)
        prev_sc = gr.Dataframe(label="预览", interactive=False)

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="指令")
    send = gr.Button("发送", variant="primary")
    chart_out = gr.Image(label="生成图表", type="filepath", height=420)


    # 加载新数据时：清空旧图表，并生成新的 thread_id
    def _up(file, name):
        info, schema, prev = handle_upload(file, name)
        path = str(PARQUET_DIR / f"{name.strip() or 'uploaded_data'}.parquet")
        if CHART_PATH.exists():
            CHART_PATH.unlink()
        # 每次上传生成全新的 uuid 作为 thread_id
        new_tid = str(uuid.uuid4())
        return info, schema, prev, name.strip() or "uploaded_data", path, None, new_tid


    # 输出增加 tid
    b_up.click(_up, [f_in, t_up], [info_up, sc, prev_up, tn, dp, chart_out, tid])


    def _sc(url, idx, name):
        info, schema, prev = handle_scrape(url, idx, name)
        path = str(PARQUET_DIR / f"{name.strip() or 'web_data'}.parquet")
        if CHART_PATH.exists():
            CHART_PATH.unlink()
        # 每次采集生成全新的 uuid 作为 thread_id
        new_tid = str(uuid.uuid4())
        return info, schema, prev, name.strip() or "web_data", path, None, new_tid


    # 输出增加 tid
    b_sc.click(_sc, [url, idx, t_sc], [info_sc, sc, prev_sc, tn, dp, chart_out, tid])

    # 输入增加 tid
    send.click(chat_fn, [msg, chatbot, tn, dp, sc, tid], [chatbot, msg, chart_out])
    msg.submit(chat_fn, [msg, chatbot, tn, dp, sc, tid], [chatbot, msg, chart_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)