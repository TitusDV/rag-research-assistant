# RAG Research Assistant

A retrieval-augmented assistant that answers questions about five well-known papers on retrieval and RAG, built so its retrieval quality can be measured rather than guessed.

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
- **Measure:** retrieval is scored with hit rate@k and mean reciprocal rank (MRR). A result counts as correct when it comes from the page a test question was written from, so different chunk sizes are compared on the same labels.

## Run it

```bash
pip install -r requirements.txt
python scripts/download_papers.py
python -m pytest
```

## Structure

```
src/research_rag/
  papers.py       the paper list
  chunking.py     loading and chunking, with paper and page labels
  retrievers.py   BM25, dense and hybrid retrievers
  metrics.py      hit rate@k and MRR
scripts/
  download_papers.py
tests/
```
