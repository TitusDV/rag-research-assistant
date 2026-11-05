# Judge check (gpt-6.1-sol)

Agreement with human review: 92% on 36 questions (Cohen's kappa 0.83; 1 is perfect, 0 is chance level).

| | Human approved | Human rejected |
|---|---|---|
| Judge approved | 17 | 1 |
| Judge rejected | 2 | 16 |

## Disagreements

- **2005.11401-p6** (you: approved, judge: duplicate): How does index hot-swapping let RAG update its knowledge compared with parametric-only models like T5 or BART?  
  Judge: The already approved question about RAG’s raw-text memory and how it can be updated covers the same index-swapping mechanism.
- **2005.11401-p2** (you: approved, judge: headline): In RAG-Token, how are retrieved documents used when generating output tokens?  
  Judge: The abstract explicitly states that one RAG formulation can use different retrieved passages per token, which is the fact this question asks about.
- **2310.11511-p0** (you: rejected, judge: ok): In SELF-RAG, what do the two categories of reflection tokens indicate?  
  Judge: The question asks for a specific distinction between reflection-token categories that is directly supported by the evidence and is not stated in the abstract or duplicated by an approved question.
