# What building jordbot taught me, and what I tried and dropped

[EVALS.md](EVALS.md) shows how jordbot is tested and what the tests decided. This page keeps the rest: what the results taught me about retrieval, prompts and checks, and the ideas I tested and rejected, each with its evidence.

Every number here was re-read from the saved results or re-computed with this repo's scripts on 2026-10-30. Ranges are 95%, resampled by question. Where a result can no longer be checked, the entry says so and gives no figure.

## Retrieval

### Retrieval follows the question's words

On 2026-09-10 I ranked every chunk an eval question expects against the whole knowledge base. 18 chunks, across 28 question-and-chunk pairs, ranked 31st or lower of 123. The material was often there. It did not use the words a recruiter uses.

Three later results show the same thing:
- **A rewrite has to answer the question.** One chunk, rewritten on 2026-10-20, ranked 7th for its question (W13). After its opening was rewritten to answer that question directly, it ranked 1st (similarity 0.671 to 0.716).
- **A sentence can carry retrieval without carrying a fact.** On 2026-10-22 I cut a sentence that stated a principle and added no fact. The chunk fell out of the top 5 for both questions that expect it (W05, C01), and all four answers failed the retrieval check. With the sentence reworded instead of cut, the chunk was back at rank 4.
- **Short, generic chunks crowd broad questions.** On a fixed 19-question probe (2026-10-14), short biography chunks took 29 of the 95 top-5 slots. Dropping two of them from the index (the text is kept) cut that to 10.

The working rule I took from this: rewording a chunk that never ranks rarely lifts it. A new chunk, written to the question, in a file that already ranks, is what has worked. The round that first showed it is not in the saved files, so treat it as practice, not a measured result.

### My name was crowding the ranking

This is the result behind two of the rejected ideas below. 13 of 145 chunks carry my name in the header that gets embedded with them: 9% of the knowledge base. 67 of the 74 eval questions name me too, so those 13 chunks matched almost every question.

Across 74 questions, they took 36% of the embedding top-3 slots with the current model (ada-002), 59% with text-embedding-3-large and 63% with 3-small. With my name taken out of the question, they took 4% to 6% on every model.

### Search noise has a size

The same 70-question retrieval check, run twice on the same index on 2026-10-30, did not return identical scores. Similarities moved by up to 0.0012 on 21 questions, and one question's top two swapped. Recall@5 was the same (0.569). MRR@5 went from 0.696 to 0.688. A change of about 0.01 in MRR between two checks is noise, not a result.

## Prompts

### A better prompt stopped citing its sources

In October I reworked the answering prompt. Citations collapsed. Over five rounds I found out why. All runs below use gpt-4o, the same 36 questions and the same knowledge base. Markers are inline source citations across the 36 answers. Follow-ups are the turns after the first in 7 multi-turn conversations.

| Run (2026-10-21 to 22) | Prompt | Source layout | Markers | Answers citing nothing | Follow-ups citing a source |
|---|---|---|---|---|---|
| Old prompt | f138f843 | tagged documents | 89 | 3 | 7 of 12 |
| New prompt | ab60e8aa | tagged documents | 48 | 11 | 1 of 12 |
| New prompt plus four citation fixes | 121b6ea8 | tagged documents | 26 | 20 | 0 of 12 |
| New prompt | ab60e8aa | id and title | 63 | 5 | 5 of 12 |
| Old prompt | f138f843 | id and title | 95 | 3 | 7 of 12 |
| New prompt, sources block moved | 9ec49116 | id and title | 65 | 6 | not run |
| New prompt, one paragraph removed | ccd205ed | id and title | 54 | 10 | not run |
| Old frame, new content (landed) | ff4d5612 | id and title | 91 and 85 (two runs) | 3 | 7 of 12 |

What I took from it:
- **The knowledge base was not the cause.** The same knowledge base under the old prompt cited 89 times.
- **More instruction made it worse.** Four edits aimed at citing produced the fewest citations of any run.
- **Layout was half of it.** The new prompt on the simpler id-and-title layout went from 48 markers to 63. The rest was the new wording.
- **Single changes did not fix the wording,** so I reversed direction. I started from the old prompt's frame and moved the decided content into it. That version cited at the old level and landed in both repos.

The three answers that cite nothing in the landed runs are the injection attacks, which have nothing to cite. Each row is one run. The two runs of the landed prompt differ by 6 markers, which gives a feel for run-to-run noise. No range was computed.

## Checks

### A passing check can mean its word list is short

The voice check (T1.6) fails an answer that narrates its sources ("the context shows") or offers more help. On 2026-10-22, reading answers by hand found phrasings it missed: "the sources do not mention", "the retrieved context", "feel free to ask", "let me know", "happy to help". I widened it twice. A prompt that looked better on voice mostly stopped looking better once the check caught all of them. The narration left is in gap and refusal answers, under every prompt tried.

### The judge passed a false gap

In release QA on 2026-10-23, the question "Has she ever owned a P&L?" was answered as a gap and then named as a weakness. My record answers it, but not in a chunk that was retrieved for that question. The grounding judge marked the answer grounded. It checks claims against what was retrieved, and nothing retrieved contradicted the gap. That defect became regression case X18 (chunk boundary) and a new chunk that answers the question.

### A forged turn reached the search query

Case M07 puts a fake assistant turn in the chat history, claiming I was a bank's CTO. On 2026-10-30, in one run of that conversation, the follow-up's rewritten search query restated the forged claim as fact. The answer then failed the injection and voice checks. The query rewrite reads the history, so it can carry a planted claim into retrieval. It is a named case and still open.

## Rejected ideas, with their evidence

### Keyword search, first version (2026-10-15)

Tested offline against the live index: 29 scored questions, top 5, the same embedded text in every arm.

| Ranking | Any expected chunk in top 5 | Recall@5 | MRR |
|---|---|---|---|
| ada-002 (live) | 24 of 29 | 0.471 | 0.702 |
| ada-002 plus BM25 keywords | 23 of 29 | 0.446 | 0.620 |
| text-embedding-3-large plus BM25 | 23 of 29 | 0.452 | 0.622 |

It traded evenly. It won on questions with concrete nouns and lost on behavioural ones. I did not adopt it. It came back on 2026-10-30 in a changed form, keeping the embedding top 3 fixed, and that version shipped ([EVALS.md](EVALS.md#keyword-search-after-the-first-version-failed)).

### Newer embedding models (2026-10-30)

I rebuilt the index from the same chunk texts with text-embedding-3-large and 3-small. Everything else was unchanged: the keyword search, the ranking, the per-file cap. 3-large was named the primary test before any result.

| Model | Recall@5 | MRR@5 | Questions with no expected chunk in top 5 |
|---|---|---|---|
| ada-002 (live) | 0.612 | 0.717 | 4 |
| text-embedding-3-large | 0.543 | 0.606 | 10 |
| text-embedding-3-small | 0.552 | 0.612 | 9 |

Against ada-002, 3-large lost 0.070 recall (−0.137 to −0.002) and 0.111 MRR (−0.198 to −0.029), both outside the noise. 3-small lost by a similar margin. Over 65 scored cases.

The cause was my name (above). The newer models gave the name-titled chunks even more weight. With the name stripped from the question, 3-large reached MRR@5 0.830 and ada-002 0.775, with recall@5 0.632 and 0.626. That comparison was not planned before the results, and the gap is small next to the cost of a reindex, so ada-002 stays. The knowledge base and the expected-source lists were built on ada-002 rankings, so this test leans towards the incumbent by an unknown amount.

### Stripping my name from the search query (2026-10-30)

Removing my name from the question before embedding it improved MRR@5 from 0.717 to 0.775 (+0.058, +0.005 to +0.113). Recall@5 barely moved (+0.008, −0.048 to +0.068).

Then I ran the full answer eval with the strip (gpt-4o, 34 questions × 3 runs). Answers got worse. Every check passed on 78% of answers, against 88% without the strip: −10 points (−20 to −2), outside the noise. All 6 lost retrieval hits were one question, asked in two cases (W05, C01).

MRR measures order inside the top 5. The answering model reads all five chunks, so order barely reaches the answer. I did not ship it. It explains the embedding result, and any future model switch has to embed without the name.

It also changed how I read the keyword-search win. Under the strip, keeping the embedding top 3 no longer beat embeddings alone: recall −0.018 (−0.057 to +0.020). Much of the keyword gain was BM25 routing round the name, since it already ignores "jordanne". Keyword search stays: it costs nothing, is not worse, and finds exact figures.

### Skipping the model when retrieval is weak

Covered in [EVALS.md](EVALS.md#what-is-deliberately-not-automated): the only true knowledge gap ranked 38th of 65 by top similarity, so no floor separates gaps from answerable questions.

### Platforms I chose not to adopt

From a review of other open-source RAG projects on 2026-10-30. I borrowed techniques, not frameworks:
- **Langfuse or Phoenix** for tracing: a second store next to the chat log I already keep.
- **Dify, Onyx, RAGFlow, DocsGPT, Open WebUI**: replacement platforms for a system that already works.
- **Knowledge-graph RAG**: too much machinery for about 150 chunks.
- **Web-search fallback**: answers must come only from my record.
