# YouTube Knowledge Intelligence System

A local, **Python-only** Retrieval-Augmented Generation (RAG) system that turns
one or more YouTube videos into a searchable, question-answerable knowledge base
— with **timestamp-aware citations** that link straight back to the exact moment
in the video.

No frontend, no backend, no web framework. Everything runs from the command line
via `python main.py`.

---

## Overview

Watching long technical videos to find one answer is slow, and generic "chat
with a video" tools tend to hallucinate and can't point you back to *where* a
claim came from. This project solves both problems:

- It ingests YouTube transcripts **with their timestamps preserved end-to-end**.
- It answers questions using **hybrid retrieval + reranking** over the transcript,
  so answers are grounded in real evidence.
- Every factual answer comes with **clickable, deep-linked citations**
  (`https://www.youtube.com/watch?v=VIDEO_ID&t=754s`) built only from the chunks
  that were actually retrieved — timestamps are never invented.
- It ships with a **real evaluation harness** that compares retrieval strategies
  on computed metrics (Recall@K, Precision@K, MRR, latency).

---

## Architecture

```
YouTube URL
     ↓
Transcript Extraction        (youtube-transcript-api)
     ↓
Timestamp Preservation       (char→time map)
     ↓
Transcript Cleaning          (cue removal, de-duplication)
     ↓
Timestamp-Aware Chunking     (sentence-packed, overlapping)
     ↓
Embedding Generation         (OpenAI | local | hash)
     ↓
Vector Database              (ChromaDB, persistent)
     ↓
Hybrid Retrieval             (Vector + BM25 via Reciprocal Rank Fusion)
     ↓
Reranking                    (identity | cross-encoder | Cohere)
     ↓
LLM                          (OpenAI-compatible; grounded prompt)
     ↓
Grounded Answer
     ↓
Clickable YouTube Timestamp Sources
```

### Project structure

```
youtube-rag/
├── app/
│   ├── ingestion/     youtube.py, transcript.py, preprocessing.py, ingest.py
│   ├── chunking/      chunker.py            (timestamp-aware chunker)
│   ├── embeddings/    embedder.py           (OpenAI / local / hash providers)
│   ├── retrieval/     base.py, vector_search.py, bm25_search.py,
│   │                  hybrid_search.py, reranker.py
│   ├── llm/           provider.py           (OpenAI / fake providers)
│   ├── rag/           prompts.py, citations.py, qa.py, topics.py
│   ├── summarization/ summarizer.py         (grounded chapters)
│   ├── learning/      quiz.py               (grounded quiz)
│   ├── evaluation/    dataset.py, evaluator.py
│   ├── storage/       vector_store.py       (Chroma / in-memory behind 1 API)
│   ├── config.py      utils.py
├── data/              transcripts/, evaluation/questions.json
├── scripts/           offline_demo.py       (no-keys end-to-end demo)
├── tests/             pytest suite (offline)
├── main.py            interactive CLI
├── requirements.txt   .env.example   .gitignore   README.md
```

---

## Features

- Single-video and **multi-video** RAG
- **Timestamp-aware citations** with clickable YouTube deep links
- **Hybrid retrieval**: semantic (vector) + keyword (BM25) fused with RRF
- **Reranking** stage (modular: identity / local cross-encoder / Cohere)
- **Grounded QA** that refuses to answer when evidence is insufficient
- **Conversation mode** with lightweight memory for follow-ups
- **Video summarization** with chapters snapped to real transcript timestamps
- **Topic search** ("where is X discussed?") with timestamped results
- **Learning mode**: study notes + transcript-grounded quiz
- **Evaluation harness** comparing Vector vs BM25 vs Hybrid vs Hybrid+Rerank
- **Persistent** knowledge base (ChromaDB) that survives restarts
- Modular providers for LLM, embeddings, reranker, and vector store
- Robust error handling, retries with backoff, and structured logging
- **Fully offline mode** (`fake` LLM + `hash` embedder) so the pipeline runs
  with no API keys — used by the test suite and the demo script

---

## Why Hybrid Retrieval?

Semantic (embedding) search understands *meaning*: it finds relevant chunks even
when the wording differs from the query. But it can under-weight exact
terminology — API names, acronyms, code identifiers — because those get smoothed
into a single dense vector. BM25 (keyword) search is the opposite: it nails exact
term matches but is blind to paraphrase.

Combining them gives the best of both. The hard part is that the two produce
scores on **totally different scales** (cosine ≈ [0,1] vs. unbounded,
corpus-dependent BM25), so naive addition lets one retriever dominate.

We use **Reciprocal Rank Fusion (RRF)** instead:

```
rrf_score(d) = Σ_retrievers  1 / (k + rank_r(d))         (k = 60 by default)
```

RRF fuses by **rank**, not raw score, so it is scale-free, needs no per-query
normalization, is robust to outliers, and rewards documents that both retrievers
agree on. In the bundled demo this is not just theory — the evaluation shows
Hybrid clearly beating either retriever alone.

## Why Reranking?

First-stage retrieval optimizes for **recall**: cast a wide net (top-15 from each
retriever) so the right chunk is *somewhere* in the candidate set. But BM25 and a
single query-vector are coarse relevance signals. A **reranker** scores each
`(query, chunk)` pair *jointly* — a much stronger signal — and keeps only the
best few. So the pattern is: **retrieve broad, rerank precise, send a small,
high-quality context to the LLM.** The reranker is modular; the default
`identity` reranker keeps the strong RRF ordering with zero extra dependencies,
and you can switch to a local cross-encoder or Cohere with one env var.

## Timestamp Citations

Timestamps are the backbone of the project and are preserved at every stage:

1. **Extraction** keeps each caption segment's `start`/`duration`.
2. **Chunking** builds a character→time map, so when sentences are packed into a
   chunk, the chunk's `start_time`/`end_time` are interpolated from the real
   caption timings — not guessed.
3. **Storage** persists `start_time`/`end_time` in chunk metadata.
4. **Citations** are built **only from retrieved chunks**, so a cited timestamp
   always corresponds to a real chunk. Chapters and quiz sources are additionally
   *snapped to the nearest actual chunk start*, making fabricated timestamps
   structurally impossible.

## Evaluation

`data/evaluation/questions.json` holds labelled questions:

```json
[
  {
    "question": "What is retrieval augmented generation?",
    "expected_video": "",
    "expected_keywords": ["retrieval", "generation", "context"]
  }
]
```

A retrieved chunk is judged **relevant** iff it belongs to an expected video
(when specified) **and** contains at least one expected keyword (when specified).
The evaluator then computes, per strategy:

- **Recall@K** — relevant retrieved in top-K / total relevant in the corpus
- **Precision@K** — relevant retrieved in top-K / K
- **HitRate@K** — share of questions with ≥1 relevant chunk in top-K
- **MRR** — mean reciprocal rank of the first relevant chunk
- **Latency** — median retrieval time (ms)

For generation (skipped for the fake LLM) it also computes answer relevance
(keyword coverage), groundedness (answer tokens found in the retrieved context),
and citation accuracy (cited timestamps that match retrieved chunks). **All
metrics are computed from real runs — nothing is hard-coded.**

Example (from `scripts/offline_demo.py`, hash embedder + tiny curated corpus):

```
 Retrieval Evaluation
============================================
Vector Search        Recall@4:  88.9%   MRR: 1.000
BM25                 Recall@4:  55.6%   MRR: 0.667
Hybrid               Recall@4: 100.0%   MRR: 1.000
Hybrid + Reranking   Recall@4: 100.0%   MRR: 1.000
```

Run it yourself: `python main.py evaluate`.

---

## Installation

Requires **Python 3.11+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat

# 2. Install dependencies (all free!)
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env        # (Windows: copy .env.example .env)
```

## ⚡ Quick Start (FREE - No Credit Card Needed!)

### Option 1: Groq API (Recommended - Fast & Free)

1. Get a free Groq API key: https://console.groq.com/
2. Edit `.env` and set `GROQ_API_KEY=your_key_here`
3. **Check available models**: https://console.groq.com/docs/models (Groq updates models frequently)
4. Run: `python main.py`

**That's it!** Everything else is already configured for free services (local embeddings, local reranking).

If you get a "model decommissioned" error, check the link above and update `LLM_MODEL` in `.env`.

### Option 2: Fully Local (No Internet, No API Keys)

1. Install **Ollama**: https://ollama.ai/
2. Start Ollama: `ollama serve` (in a separate terminal)
3. Pull a model: `ollama pull mistral`
4. Edit `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   LLM_MODEL=mistral
   EMBEDDING_PROVIDER=local
   RERANKER_PROVIDER=cross_encoder
   ```
5. Run: `python main.py`

### Option 3: Offline Demo (No Setup Needed)

```bash
python scripts/offline_demo.py          # full pipeline on canned transcripts, no keys needed
```

---

## Configuration

All configuration lives in `.env` (loaded by `python-dotenv`) and is read into a
single `Config` class — **no secrets are ever hard-coded**. Key variables:

### LLM Provider

| Option | Provider | Cost | Speed | Setup |
|---|---|---|---|---|
| **groq** ⭐ | Groq API | FREE (free tier) | Very Fast | Get key at https://console.groq.com/ |
| **ollama** | Local (Ollama) | FREE (local) | Medium | Install Ollama + pull model |
| **openai** | OpenAI | PAID | Fast | Requires credit card |
| **fake** | Deterministic offline | FREE (local) | Instant | No setup (for testing) |

### Embeddings

| Option | Provider | Cost | Speed | Quality |
|---|---|---|---|---|
| **local** ⭐ | Sentence-transformers | FREE (local) | Medium | Good (~384-dim) |
| **openai** | OpenAI API | PAID | Fast | Excellent |
| **hash** | Deterministic offline | FREE (local) | Fast | Poor (deterministic) |

### Reranking

| Option | Provider | Cost | Setup |
|---|---|---|---|
| **cross_encoder** ⭐ | Local cross-encoder | FREE (local) | Included |
| **identity** | No reranking | FREE (local) | Included |
| **cohere** | Cohere API | PAID | Requires API key |

### Complete Configuration Reference

| Variable | Options | Default |
|---|---|---|
| `LLM_PROVIDER` | `groq`, `ollama`, `openai`, `fake` | `groq` |
| `GROQ_API_KEY` | Your key from console.groq.com | — |
| `OLLAMA_BASE_URL` | e.g., `http://localhost:11434` | — |
| `LLM_MODEL` | Groq: `mixtral-8x7b-32768`, `llama2-70b-4096` | `mixtral-8x7b-32768` |
| | Ollama: `mistral`, `neural-chat`, `llama2` | — |
| `EMBEDDING_PROVIDER` | `local`, `openai`, `hash` | `local` |
| `EMBEDDING_MODEL` | For local: model name on HuggingFace | `all-MiniLM-L6-v2` |
| `RERANKER_PROVIDER` | `cross_encoder`, `identity`, `cohere` | `cross_encoder` |
| `VECTOR_STORE` | `chroma`, `memory` | `chroma` |
| `TOP_K`, `VECTOR_TOP_K`, `BM25_TOP_K`, `RERANK_TOP_K` | Integer | 5/15/15/5 |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Characters | 800/150 |
| `MAX_RETRIES`, `RETRY_BACKOFF`, `REQUEST_TIMEOUT` | Tuning | 3/1.5/60 |

---

## Cost Comparison

| Setup | LLM Cost | Embeddings Cost | Reranking Cost | Total |
|---|---|---|---|---|
| **Default (Groq + Local)** | $0 (free tier) | $0 | $0 | **$0** ✅ |
| Ollama (Local) | $0 | $0 | $0 | **$0** ✅ |
| OpenAI (Full) | $$ | $$$ | $$ | **$$$$$ (expensive)** ❌ |

---

## Usage

```
python main.py            # interactive menu
python main.py evaluate   # run the evaluation and print the report
```

Interactive menu:

```
1. Add YouTube video      # paste a URL; transcript → chunks → embeddings → Chroma
2. List indexed videos
3. Ask a question         # conversation mode with grounded, cited answers
4. Search topics          # find where a topic is discussed (timestamped)
5. Summarize video        # overview + real-timestamp chapters
6. Compare videos         # multi-video comparison, grouped + cited
7. Learning mode          # study notes + grounded quiz
8. Run evaluation         # strategy comparison + metrics
9. Exit
```

**Try it without any API keys:**

```bash
python scripts/offline_demo.py          # full pipeline on canned transcripts
```

---

## Design decisions worth calling out

- **Custom timestamp-aware chunker instead of LangChain splitters.**
  Off-the-shelf splitters flatten text to a plain string and would destroy the
  per-word timing that makes citations possible. We build a character→time map
  and pack whole sentences into overlapping chunks, deriving each chunk's start/
  end from the real caption timings. (This is the main place LangChain would
  normally be used; here a bespoke component is the correct engineering call, so
  the core RAG logic stays explicit and inspectable rather than hidden behind
  abstractions.)
- **Provider abstractions everywhere** (LLM, embeddings, reranker, vector store)
  so components are swappable and the whole thing is testable offline.
- **Storage behind one interface** so replacing Chroma with PostgreSQL/pgvector
  later is a localized change.

## Limitations

- Depends on YouTube having a transcript/captions available for the video.
- Auto-generated captions can be noisy; cleaning helps but isn't perfect.
- The default `identity` reranker preserves RRF order rather than doing joint
  scoring; enable the cross-encoder or Cohere reranker for best quality.
- The `hash` embedder is deterministic and dependency-free but not competitive
  with real embedding models — it exists for offline tests/demos. Use the OpenAI
  or local embedder for real work.
- Generation-quality metrics (relevance, groundedness) are lightweight proxies,
  not a full LLM-graded evaluation.
- No speaker diarization; transcripts are treated as a single stream.

## Future Improvements

- Agentic research mode (multi-step, tool-using retrieval)
- Knowledge graph over entities/topics across videos
- PostgreSQL/pgvector backend for the vector store
- Local LLM support (e.g. via an OpenAI-compatible local server)
- Multilingual transcripts and translation
- Speaker identification / diarization

---

## Running the tests

```bash
pip install -r requirements.txt
pytest -q
```

The suite (46 tests) runs **fully offline** — it injects a fake transcript
provider, the deterministic hash embedder, and a fake LLM, and it exercises a
real ChromaDB persistence round-trip in a temp directory.

---

### Resume-ready project description

> Built an end-to-end, production-minded **RAG system over YouTube transcripts**
> in pure Python: timestamp-preserving ingestion and semantic chunking, a
> ChromaDB vector store, **hybrid retrieval (dense + BM25) fused with Reciprocal
> Rank Fusion**, a modular reranking stage, and grounded LLM answers with
> **clickable timestamp citations**. Added summarization with real-timestamp
> chapters, multi-video comparison, topic search, a transcript-grounded quiz
> generator, and a **quantitative evaluation harness** (Recall@K, Precision@K,
> MRR, latency) that benchmarks retrieval strategies. Fully modular providers,
> offline test mode, and a 46-test pytest suite.

### Suggested GitHub repository description

> Pure-Python, CLI RAG system for YouTube videos: timestamp-aware chunking,
> hybrid (vector + BM25) retrieval with RRF, reranking, grounded answers with
> clickable timestamp citations, summaries, quizzes, and a real evaluation harness.
