# RAG Research Assistant

![tests](https://github.com/TitusDV/rag-research-assistant/actions/workflows/tests.yml/badge.svg)

A question-answering assistant over five research papers about retrieval and RAG. My earlier [Telecare RAG chatbot](https://github.com/TitusDV/telecare-rag-chatbot) used hybrid retrieval but I never measured whether it actually found the right passages. This project uses the same approach and this time I measured it.

The papers:

| arXiv | Paper |
|---|---|
| 2005.11401 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| 2004.04906 | Dense Passage Retrieval for Open-Domain Question Answering |
| 2307.03172 | Lost in the Middle: How Language Models Use Long Contexts |
| 2309.15217 | Ragas: Automated Evaluation of Retrieval Augmented Generation |
| 2310.11511 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection |

The PDFs are downloaded from arXiv when you run it. They are not in the repo because arXiv's licence doesn't allow redistributing them.

## How I tested retrieval

Every page of every paper is labelled with its arXiv id and page number, and those labels stay with the chunks after splitting. I then needed questions where I knew which page had the answer. A model (gpt-6-luna) read sampled pages and wrote one question per page, along with an exact quote from the page as evidence. If the quote wasn't really on the page, the question was thrown out.

The questions still needed checking. A lot of them were bad in ways I didn't expect: questions about citations in the reference list, about which table shows what, about trivia inside worked examples, or about the paper's main claim, which appears on several pages so there is no single right answer. I reviewed the first batch by hand, then wrote an LLM judge that follows the same rules. I ran the check twice with the same judge model (gpt-6.1-sol). It agreed with my decisions on 34 of 36 questions the first time (Cohen's kappa 0.89) and 33 of 36 the second (kappa 0.83); the latest run is in `eval/judge_check.md`. Kappa corrects for agreement that would happen by chance, and above 0.8 is usually read as almost perfect. The one-question difference between runs is a reminder that an LLM judge isn't fully repeatable. I let it review the second batch.

Each retriever (BM25, dense embeddings, and a hybrid of the two) was run on whole pages and on 1000, 500 and 250 character chunks. A result counts as correct if it comes from the question's page. Scoring by page instead of by chunk means the answer key doesn't change when the chunk size does.

### The measures

- **Hit@k**: share of questions where a passage from the right page is in the top k results. Hit@6 matches the app, which gives the model six passages.
- **Evidence@5 and Evidence@6**: stricter. The passage must be from the right page and contain the question's whole evidence quote word for word, after normalising case, spacing, line-break hyphens and PDF ligatures, and the match must start and end on whole words, so "80% accuracy" doesn't match inside "180% accuracy". A passage from the right page but the wrong paragraph doesn't count. It undercounts when a chunk boundary splits the quote, which happens more often with small chunks: the quotes are up to 30 words, close to the length of a 250-character chunk. A matching quote shows the right paragraph was found, not that the passage fully supports an answer.

  An earlier version matched on shared words and could credit a passage containing only "the score". It was replaced after a review, and every Evidence figure here comes from the corrected rule. The stricter check also caught two quotes that ended where the PDF text has a footnote number stuck to the last word ("relevance.4" and "20225"). I trimmed those two quotes so they end before the footnote; the questions and pages are unchanged.
- **MRR@10**: on average, how high the first right passage ranks (1 means always first).
- **nDCG@10**: like MRR, but a passage at rank 2 or 3 still earns most of the credit. That fits the app, which reads six passages.
- **95% intervals**: each score is shown with a bootstrap interval, resampling the questions. With 36 questions they are wide, which is the honest picture.
- **Paired comparison**: every setup is compared with the one the app uses, question by question. If the interval for the difference includes 0, the data can't tell the two apart.

I left out precision@k. With one right page per question, a whole-page retriever can score at most 1 in 10 while small chunks can fill the top 10 from the same page, so it can't be compared fairly across chunk sizes.

### Reranking

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) can reread the hybrid retriever's top 20 passages together with the question and reorder them. It reads question and passage at the same time, so it is usually better at putting the best passage first, at the cost of extra time per question. `python scripts/run_eval.py --rerank` scores it next to the other setups, including time per question.

On these questions it didn't earn its place: see the results below.

### Answer quality

Retrieval scores say whether the right passage was found, not whether the user got a correct answer. For that, `scripts/run_answer_eval.py` runs the assistant exactly as the app does on the held-out questions, and saves for each one the passages the model saw (in rank order), the model's raw answer and citation numbers, and the final answer with its sources. A person then fills in a review sheet, and `scripts/score_answers.py` reports:

- **Answer correctness**: answerable questions where the answer has every required fact with no material error. Declining an answerable question counts as a miss, and is also reported on its own.
- **Citation support**: factual claims fully supported by the passage they cite, including numbers and comparisons.
- **Correct abstention**: questions the papers can't answer that the assistant declined.
- **Why failures happened**: whether the right page was among the passages (a generation problem) or not (a retrieval problem).

Run errors such as timeouts are kept apart from deliberate refusals.

## Results

36 questions. The table shows how often the right page was in the top 5 results, for every retriever and passage size. The full table, with every measure and the paired comparisons, is in `eval/results.md`.

| Retriever | Whole page | 1000 chars | 500 chars | 250 chars |
|---|---|---|---|---|
| BM25 (keyword) | 33/36 | **35/36** | 32/36 | 33/36 |
| Hybrid | 30/36 | 32/36 | **34/36** (used in the app) | 32/36 |
| Dense (embeddings) | 19/36 | 27/36 | 30/36 | 29/36 |

Ranking, passage-level evidence and reranking, for the setup the app uses and its closest rivals (95% intervals in brackets):

| Setup | Hit@1 | Hit@5 | Evidence@5 | Evidence@6 | MRR@10 | ms per question |
|---|---|---|---|---|---|---|
| Hybrid, 500 (app) | 0.61 | 0.94 (0.86-1.00) | 0.61 | 0.61 | 0.750 (0.64-0.85) | 30 |
| Hybrid + rerank, 500 | 0.69 | 0.89 (0.78-0.97) | 0.58 | 0.61 | 0.777 (0.66-0.88) | 869 |
| BM25, 1000 | 0.69 | 0.97 (0.92-1.00) | 0.83 | 0.83 | 0.788 (0.68-0.89) | 2 |
| Hybrid, 1000 | 0.56 | 0.89 (0.78-0.97) | 0.78 | 0.81 | 0.701 (0.59-0.82) | 26 |

What I took from it:

- No setup is clearly better than the app's. Compared question by question, no setup's difference from it has an interval entirely above zero. The clearly worse ones are dense retrieval and hybrid on whole pages, with or without reranking.
- The app passes six passages to the model, so what matters is whether the right page is among the top few, not whether it comes first. Hybrid on 500 characters got 34 of 36 into the top 5, and the sixth passage added none (34 of 36 in the top 6).
- The reranker moved more right passages to first place (Hit@1 0.61 to 0.69) but dropped some out of the top 5, and the overall change could be chance. It also made retrieval about 30 times slower (30 ms to 869 ms per question on my laptop). I left it out of the app. Reranking whole pages made results clearly worse, probably because the cross-encoder only reads the start of a long passage, so most of each page is never scored.
- Evidence@5 shows a trade-off the page-level scores hide: with 500-character passages, the passage holding the evidence quote was in the top 5 for 61% of questions, against 78% with 1000-character passages and 42% with 250. Part of that gap is the measure itself, because a small chunk is more likely to cut the quote in two. On the held-out questions the pattern didn't hold (0.60 for 500 against 0.50 for 1000), so the app stays at 500.
- BM25 probably does better here than it would with real users, because the generated questions tend to reuse the page's own wording. That is why the app keeps the dense half.
- Dense retrieval on whole pages was poor, and it got much better as chunks got smaller. The embedding model (all-MiniLM-L6-v2) only reads the first 256 tokens of its input, so on a full page most of the text is never embedded.

### Held-out check

Ten AI-assisted questions, two per paper, each evidence quote checked against the source PDF, and kept out of choosing the app's setup (`eval/results_heldout.md`). Before the rerun, a validation script checks every question's paper, page and quote (`python scripts/validate_heldout.py`). With ten questions every interval is wide, so this is a check on the main results, not a new ranking.

| Setup | Hit@1 | Hit@5 | Evidence@5 | MRR@10 |
|---|---|---|---|---|
| Hybrid, 500 (app) | 0.70 | 8/10 | 0.60 | 0.750 (0.45-1.00) |
| Hybrid + rerank, 500 | 0.80 | 10/10 | 0.80 | 0.900 (0.75-1.00) |
| BM25, 1000 | 0.30 | 8/10 | 0.70 | 0.470 (0.25-0.70) |

- BM25 on 1000 characters, joint best on the original questions, ranked clearly worse than the app's setup here (MRR difference -0.28, interval -0.53 to -0.05). Its relative performance changed between the two question sets. Wording overlap between generated questions and their pages is one possible explanation, but both sets were AI-assisted and differ in content, so this doesn't prove the cause.
- The reranker, which didn't help on the original questions, found the right page for all ten here. The difference from the app's setup could still be chance (both intervals start at 0), and the two sets point different ways, so it stays out of the demo until a bigger held-out set settles it.
- The app's setup found the right page in the top 5 for 8 of 10, against 34 of 36 on the original questions. Some drop is expected on questions that played no part in choosing it.

## Asking questions

`python app.py` opens a small web app. The assistant uses hybrid retrieval on 500 character chunks, for the reasons above. It answers only from the passages it retrieves, cites them by paper and page, and says "The papers don't cover this." when they don't. Every citation in the answer is checked against the retrieved passages: markers that don't match a passage are dropped, each cited page gets one number, and an answer with no valid citation is treated as not covered. Each source shows the passage it came from. It downloads the papers itself on first start, so it also runs as a Hugging Face Space.

## Running it

```bash
pip install -r requirements.txt
python scripts/download_papers.py
python -m pytest

cp .env.example .env                    
python scripts/make_questions.py        # wmake questions
python scripts/review_questions.py      # review by hand
python scripts/auto_review.py --check   
python scripts/auto_review.py          
python scripts/run_eval.py              # score every retriever and chunk size
python scripts/run_eval.py --rerank     # ...and hybrid reranked by a cross-encoder
python scripts/validate_heldout.py      # check the held-out questions' papers, pages and quotes
python scripts/run_eval.py --heldout    # score the held-out questions
python scripts/run_answer_eval.py       # answer the held-out questions, save passages and a review sheet
python scripts/score_answers.py         # score the filled-in review sheet
python scripts/check_injection.py       # try hidden instructions inside passages against the model
python app.py                           
```

## Code

```
src/research_rag/
  papers.py       the five papers
  chunking.py     loading and chunking with paper and page labels
  retrievers.py   BM25, dense and hybrid
  questions.py    question generation, evidence check, reference-page filter
  review.py       the LLM judge
  evaluate.py     the comparison
  answer.py       answering with citations
  ui.py           the web app's answer panel and daily question limit
  metrics.py      hit rate, MRR, nDCG, bootstrap intervals, Cohen's kappa
  rerank.py       cross-encoder reranking
  answer_eval.py  answer correctness, citation support and abstention on held-out questions
  llm.py          OpenAI or Ollama model
app.py, app.css   the web app
space/            the Hugging Face Space card
scripts/          the commands above, plus check_reference_filter.py
eval/             the questions, results (with per-question ranks in retrieval_details*.json) and judge check
tests/
```
