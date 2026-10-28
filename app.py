"""A small web app for asking the papers questions.  Run: python app.py"""
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from research_rag.answer import Assistant, make_answerer  # noqa: E402
from research_rag.chunking import load_papers  # noqa: E402
from research_rag.llm import build_chat_model  # noqa: E402

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
load_dotenv(ROOT / ".env")

print("Loading papers and building the index (takes a minute) ...")
assistant = Assistant(
    load_papers(ROOT / "data" / "papers"),
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    make_answerer(build_chat_model(os.getenv("MODEL", "llama3"))),
)


def respond(question: str) -> tuple[str, str]:
    if not question.strip():
        return "", ""
    result = assistant.ask(question)
    sources = "\n".join(f"[{s['n']}] {s['title']}, page {s['page']}  (arxiv.org/abs/{s['paper']})"
                        for s in result["sources"])
    return result["answer"], sources


with gr.Blocks(title="RAG Research Assistant") as demo:
    gr.Markdown("# RAG Research Assistant\nAsk about RAG, Dense Passage Retrieval, Lost in the Middle, Ragas or Self-RAG.")
    question = gr.Textbox(label="Question", placeholder="Why do language models miss information in the middle of long contexts?")
    ask = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.Textbox(label="Sources", lines=4)
    ask.click(respond, inputs=question, outputs=[answer, sources])
    question.submit(respond, inputs=question, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()
