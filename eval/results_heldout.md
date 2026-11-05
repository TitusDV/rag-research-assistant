# Retrieval results on held-out questions

AI-assisted questions with source-checked reference quotes, originally reserved from setup selection. Subsequent reuse should be reported.
Embedding model: sentence-transformers/all-MiniLM-L6-v2

Questions: 10. Results are cut to the top 10. Brackets: 95% bootstrap interval.

Evidence@k is the fraction with the complete normalised reference quote in one top-k passage on the labelled page. Normalisation uses NFKC, line-end dehyphenation, whitespace and case. Matches respect word and number boundaries. Split quotes and equivalent wording can be missed.

| Retriever | Chunk size | Chunks | Hit@1 | Hit@3 | Hit@5 | Hit@6 | Evidence@5 | Evidence@6 | MRR@10 | nDCG@10 | ms/question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid+rerank | 500 | 759 | 0.80 | 1.00 | 1.00 (1.00-1.00) | 1.00 | 0.80 | 0.80 | 0.900 (0.75-1.00) | 0.926 | 771 |
| hybrid | 250 | 1574 | 0.70 | 0.80 | 0.80 (0.50-1.00) | 0.80 | 0.40 | 0.40 | 0.761 (0.50-0.95) | 0.793 | 35 |
| hybrid | 500 | 759 | 0.70 | 0.80 | 0.80 (0.50-1.00) | 0.80 | 0.60 | 0.60 | 0.750 (0.45-1.00) | 0.763 | 27 |
| hybrid+rerank | 1000 | 402 | 0.60 | 0.80 | 0.90 (0.70-1.00) | 0.90 | 0.60 | 0.60 | 0.708 (0.45-0.93) | 0.756 | 1415 |
| hybrid+rerank | 250 | 1574 | 0.50 | 0.80 | 0.90 (0.70-1.00) | 0.90 | 0.40 | 0.40 | 0.642 (0.40-0.87) | 0.706 | 446 |
| bm25 | 500 | 759 | 0.50 | 0.70 | 0.80 (0.50-1.00) | 0.80 | 0.60 | 0.60 | 0.634 (0.38-0.87) | 0.698 | 4 |
| dense | 250 | 1574 | 0.50 | 0.60 | 0.80 (0.50-1.00) | 0.80 | 0.40 | 0.40 | 0.583 (0.30-0.83) | 0.636 | 27 |
| bm25 | 250 | 1574 | 0.30 | 0.60 | 0.90 (0.70-1.00) | 0.90 | 0.40 | 0.50 | 0.492 (0.28-0.72) | 0.592 | 7 |
| bm25 | page | 88 | 0.40 | 0.40 | 0.60 (0.30-0.90) | 0.60 | 0.60 | 0.60 | 0.472 (0.21-0.74) | 0.547 | 2 |
| hybrid | 1000 | 402 | 0.30 | 0.60 | 0.60 (0.30-0.90) | 0.70 | 0.50 | 0.70 | 0.472 (0.25-0.72) | 0.572 | 24 |
| bm25 | 1000 | 402 | 0.30 | 0.70 | 0.80 (0.50-1.00) | 0.80 | 0.70 | 0.70 | 0.470 (0.25-0.70) | 0.552 | 3 |
| dense | 500 | 759 | 0.10 | 0.70 | 0.80 (0.50-1.00) | 0.80 | 0.60 | 0.60 | 0.383 (0.23-0.56) | 0.510 | 26 |
| hybrid | page | 88 | 0.10 | 0.40 | 0.50 (0.20-0.80) | 0.50 | 0.50 | 0.50 | 0.262 (0.10-0.45) | 0.365 | 23 |
| dense | 1000 | 402 | 0.00 | 0.40 | 0.60 (0.30-0.90) | 0.60 | 0.40 | 0.40 | 0.228 (0.10-0.35) | 0.321 | 21 |
| hybrid+rerank | page | 88 | 0.10 | 0.30 | 0.30 (0.10-0.60) | 0.30 | 0.30 | 0.30 | 0.223 (0.06-0.42) | 0.310 | 2278 |
| dense | page | 88 | 0.10 | 0.20 | 0.20 (0.00-0.50) | 0.20 | 0.20 | 0.20 | 0.160 (0.00-0.36) | 0.192 | 21 |

## Compared with hybrid/500 (the setup the app uses)

Difference = other setup minus this one, on the same questions. If the interval includes 0, the difference could be chance.

| Setup | Hit@5 difference | MRR difference | Clear difference? |
|---|---|---|---|
| bm25/page | -0.20 (-0.60 to +0.20) | -0.277 (-0.60 to +0.04) | no |
| dense/page | -0.60 (-0.90 to -0.30) | -0.590 (-0.89 to -0.29) | Hit@5, MRR |
| hybrid/page | -0.30 (-0.60 to +0.00) | -0.488 (-0.75 to -0.21) | MRR |
| hybrid+rerank/page | -0.50 (-0.80 to -0.20) | -0.527 (-0.81 to -0.23) | Hit@5, MRR |
| bm25/1000 | +0.00 (-0.30 to +0.30) | -0.280 (-0.53 to -0.05) | MRR |
| dense/1000 | -0.20 (-0.50 to +0.00) | -0.522 (-0.73 to -0.30) | MRR |
| hybrid/1000 | -0.20 (-0.50 to +0.00) | -0.277 (-0.51 to -0.07) | MRR |
| hybrid+rerank/1000 | +0.10 (+0.00 to +0.30) | -0.042 (-0.20 to +0.07) | no |
| bm25/500 | +0.00 (+0.00 to +0.00) | -0.116 (-0.34 to +0.10) | no |
| dense/500 | +0.00 (+0.00 to +0.00) | -0.368 (-0.56 to -0.16) | MRR |
| hybrid+rerank/500 | +0.20 (+0.00 to +0.50) | +0.150 (+0.00 to +0.40) | no |
| bm25/250 | +0.10 (+0.00 to +0.30) | -0.258 (-0.53 to +0.03) | no |
| dense/250 | +0.00 (-0.30 to +0.30) | -0.167 (-0.42 to +0.03) | no |
| hybrid/250 | +0.00 (-0.30 to +0.30) | +0.011 (-0.27 to +0.30) | no |
| hybrid+rerank/250 | +0.10 (+0.00 to +0.30) | -0.108 (-0.38 to +0.18) | no |
