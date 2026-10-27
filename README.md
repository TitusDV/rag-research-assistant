# RAG Research Assistant

A retrieval-augmented assistant that answers questions about five well-known papers on retrieval and RAG, built so its retrieval quality can be measured rather than guessed.

You will see that I have used the same method that was used in telecare rag project

## Papers

| arXiv | Paper |
|---|---|
| 2005.11401 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| 2004.04906 | Dense Passage Retrieval for Open-Domain Question Answering |
| 2307.03172 | Lost in the Middle: How Language Models Use Long Contexts |
| 2309.15217 | Ragas: Automated Evaluation of Retrieval Augmented Generation |
| 2310.11511 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection |

The PDFs are downloaded from arXiv at runtime and are not stored in this repo.

## How it works

- **Load:** each paper is read page by page, and every page is labelled with its arXiv id and page number.
- **Chunk:** pages are split with a recursive character splitter; chunks keep their paper and page labels.
- **Retrieve:** three retrievers over the same chunks: BM25 keyword search, dense embedding search, and a hybrid of the two.
- **Test questions:** a model reads sampled pages (the same number from each paper) and writes one question per page, with a short exact quote as evidence. Questions whose quote is not actually on the page are dropped automatically. The rest are reviewed against a checklist (no citations, no layout or example trivia, no headline claims from the abstract, no duplicates, no answers beyond the evidence), by hand or by an LLM judge whose agreement with human review is measured (`eval/judge_check.md`).
- **Compare:** BM25, dense and hybrid retrieval are each run on whole pages and on 1000, 500 and 250 character chunks, and scored on the approved questions.
- **Measure:** retrieval is scored with hit rate@k and mean reciprocal rank (MRR). A result counts as correct when it comes from the page a test question was written from, so different chunk sizes are compared on the same labels.

## Results

36 reviewed questions, top 10 results per retriever (full table: eval/results.md).

Retriever	Chunk size	Hit@1	Hit@5	MRR@10
BM25	whole page	0.72	0.92	0.788
BM25	1000	0.69	0.97	0.788
Hybrid	250	0.69	0.89	0.768
Dense	250	0.61	0.81	0.688
Dense	whole page	0.22	0.53	0.359
BM25 and hybrid retrieval are effectively tied at the top; with 36 questions, gaps under about 0.03 are within one question.
Dense retrieval improves sharply as chunks shrink, because all-MiniLM-L6-v2 reads at most 256 tokens and ignores the rest of a longer chunk.
BM25 likely benefits from the generated questions reusing each page’s wording, so these results favour keyword search more than real user questions would.
The LLM judge agreed with human review on 94% of 36 questions.

## Run these

cp .env.example .env              # choose the model
python scripts/make_questions.py  # writes eval/questions.jsonl
python scripts/review_questions.py   # by hand:
python scripts/auto_review.py --check  # measure the judge against revews
python scripts/run_eval.py        # writes eval/results.md
```

## Structure

```
src/research_rag/
  papers.py       the paper list
  chunking.py     loading and chunking, with paper and page labels
  retrievers.py   BM25, dense and hybrid retrievers
  metrics.py      hit rate@k and MRR
  questions.py    test-question generation and the evidence check
  llm.py          Ollama or OpenAI chat model
  review.py       the LLM judge and its agreement check
  evaluate.py     scores every retriever and chunk size
scripts/
  download_papers.py
  make_questions.py
  review_questions.py
  auto_review.py
  check_reference_filter.py
  run_eval.py
eval/
  questions.jsonl the reviewed test set
  results.md      the comparison table
tests/
```
