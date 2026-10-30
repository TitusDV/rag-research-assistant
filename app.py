"""The web app for asking the papers questions.  Run: python app.py

Also runs as a Hugging Face Space (see space/README.md).
"""
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:  # only installed on Hugging Face ZeroGPU Spaces; must be imported before torch
    import spaces
    on_gpu = spaces.GPU(duration=10)
except ImportError:
    def on_gpu(fn):
        return fn

import gradio as gr  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from research_rag.answer import Assistant, make_answerer  # noqa: E402
from research_rag.chunking import load_papers  # noqa: E402
from research_rag.llm import build_chat_model  # noqa: E402
from research_rag.papers import download  # noqa: E402
from research_rag.ui import DailyLimit, render_empty, render_message, render_papers, render_result  # noqa: E402

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
load_dotenv(ROOT / ".env")

PAPER_DIR = ROOT / "data" / "papers"
download(PAPER_DIR)
pages = load_papers(PAPER_DIR)
print(f"loaded {len(pages)} pages, {sum(len(p.page_content) for p in pages)} characters of text")
print("Building the index (takes a minute) ...")
assistant = Assistant(
    pages,
    # CPU on purpose: on ZeroGPU, torch reports a (simulated) GPU outside @spaces.GPU functions.
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}),
    make_answerer(build_chat_model(os.getenv("MODEL", "llama3"))),
)
limit = DailyLimit(int(os.getenv("DAILY_LIMIT", "100")))

EXAMPLES = [
    "What happens when the answer sits in the middle of a long context?",
    "How does Self-RAG decide when to retrieve?",
    "What does Ragas measure without human labels?",
    "How is Dense Passage Retrieval trained?",
]


def left_text() -> str:
    return f"{limit.remaining()} questions left today"


@on_gpu
def gpu_check() -> None:
    """Never called. A ZeroGPU Space won't start without at least one @spaces.GPU function,
    but this app doesn't need a GPU: retrieval and the small embedding model run fine on CPU."""


def respond(question: str):
    question = question.strip()
    if not question:
        return render_empty(), left_text()
    if not limit.take():
        return render_message("The daily limit for this demo has been reached. Try again tomorrow."), left_text()
    try:
        result = assistant.ask(question)
        print(f"asked: {question!r} answered={result['answered']} "
              f"sources={[(s['paper'], s['page']) for s in result['sources']]}")
    except Exception as err:  # keep the page usable if the model call fails
        print(f"error: {err!r}")
        return render_message("Something went wrong getting an answer. Please try again."), left_text()
    return render_result(result), left_text()


THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(c50="#EEF6F7", c100="#D6E7EA", c200="#B3D3D9", c300="#86B8C1", c400="#4F96A3",
                                c500="#0F6B7E", c600="#0E5E6F", c700="#0B4C5A", c800="#093C47", c900="#0D2A30",
                                c950="#081B1F"),
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Schibsted Grotesk", weights=(400, 500, 600, 700, 800)), "system-ui", "sans-serif"],
    radius_size="md",
).set(
    body_background_fill="#F5FAFA", body_background_fill_dark="#F5FAFA",
    body_text_color="#0D2A30", body_text_color_dark="#0D2A30",
    block_background_fill="#FFFFFF", block_background_fill_dark="#FFFFFF",
    block_border_color="#D6E7EA", block_border_color_dark="#D6E7EA",
    input_background_fill="#FFFFFF", input_background_fill_dark="#FFFFFF",
    border_color_primary="#D6E7EA", border_color_primary_dark="#D6E7EA",
    button_primary_background_fill="#0E5E6F", button_primary_background_fill_dark="#0E5E6F",
    button_primary_background_fill_hover="#0B4C5A", button_primary_background_fill_hover_dark="#0B4C5A",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
)

CSS = (ROOT / "app.css").read_text(encoding="utf-8")

with gr.Blocks(title="RAG Research Assistant") as demo:
    gr.HTML(
        '<header class="top"><span class="mark">t.</span><div><h1>RAG Research Assistant</h1>'
        "<p>Answers questions from five research papers on retrieval, and shows the page each answer came from.</p>"
        '</div><a class="repo" href="https://github.com/TitusDV/rag-research-assistant" target="_blank" '
        'rel="noopener">Code on GitHub</a></header>'
    )
    with gr.Row(equal_height=False, elem_classes="layout"):
        with gr.Column(scale=3, elem_classes="main"):
            with gr.Row(elem_classes="ask-row"):
                question = gr.Textbox(show_label=False, container=False, scale=5, autofocus=True, max_length=400,
                                      placeholder="Ask about RAG, DPR, Lost in the Middle, Ragas or Self-RAG")
                ask = gr.Button("Ask", variant="primary", scale=1, min_width=90)
            with gr.Row(elem_classes="chips"):
                chips = [gr.Button(text, size="sm", elem_classes="chip") for text in EXAMPLES]
            answer = gr.HTML(render_empty())
        with gr.Column(scale=2, elem_classes="side"):
            gr.HTML('<p class="label">Papers covered</p>' + render_papers())
            gr.HTML('<p class="label">How it works</p><p class="how">Each question is matched against '
                    "500 character passages using keyword search and embeddings together. A language model then "
                    "answers using only the top passages, citing each one, and says so when they don't contain "
                    "the answer.</p>")
            left = gr.Markdown(left_text(), elem_classes="left")

    ask.click(respond, inputs=question, outputs=[answer, left])
    question.submit(respond, inputs=question, outputs=[answer, left])
    for chip in chips:
        chip.click(lambda text: text, inputs=chip, outputs=question).then(respond, inputs=question, outputs=[answer, left])
    demo.load(left_text, outputs=left)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
