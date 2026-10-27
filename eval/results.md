# Retrieval results

Embedding model: sentence-transformers/all-MiniLM-L6-v2

Approved questions: 36. Results are cut to the top 10.

| Retriever | Chunk size | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | page | 88 | 0.72 | 0.78 | 0.92 | 0.788 |
| bm25 | 1000 | 402 | 0.69 | 0.86 | 0.97 | 0.788 |
| hybrid | 250 | 1574 | 0.69 | 0.81 | 0.89 | 0.768 |
| bm25 | 500 | 759 | 0.67 | 0.83 | 0.89 | 0.760 |
| hybrid | 500 | 759 | 0.61 | 0.86 | 0.94 | 0.750 |
| bm25 | 250 | 1574 | 0.58 | 0.86 | 0.92 | 0.718 |
| hybrid | 1000 | 402 | 0.56 | 0.81 | 0.89 | 0.701 |
| dense | 250 | 1574 | 0.61 | 0.72 | 0.81 | 0.688 |
| dense | 1000 | 402 | 0.47 | 0.64 | 0.75 | 0.594 |
| dense | 500 | 759 | 0.39 | 0.69 | 0.83 | 0.573 |
| hybrid | page | 88 | 0.31 | 0.67 | 0.83 | 0.522 |
| dense | page | 88 | 0.22 | 0.42 | 0.53 | 0.359 |
