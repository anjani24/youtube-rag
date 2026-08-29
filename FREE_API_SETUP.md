# Free API Setup Guide

This guide walks you through setting up the YouTube RAG project with **completely free** APIs and services. No credit card required!

## Table of Contents
1. [Option 1: Groq API (Recommended)](#option-1-groq-api-recommended)
2. [Option 2: Ollama (Fully Local)](#option-2-ollama-fully-local)
3. [Option 3: Offline Demo](#option-3-offline-demo)
4. [Troubleshooting](#troubleshooting)

---

## Option 1: Groq API (Recommended)

**Best for:** Most users. Fast, free, requires only an API key.

### Step 1: Get a Groq API Key

1. Go to https://console.groq.com/
2. Sign up (free - no credit card needed)
3. Navigate to **API Keys** section
4. Create a new API key
5. Copy the key

### Step 2: Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` and find the LLM section:

```env
# RECOMMENDED: Groq API (FREE, no credit card, very fast)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_YOUR_KEY_HERE_xxxxx

# Leave these as-is (all free):
EMBEDDING_PROVIDER=local
RERANKER_PROVIDER=cross_encoder
```

### Step 3: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install
pip install -r requirements.txt
```

### Step 4: Run

```bash
python main.py
```

You'll see a menu:
```
1. Add YouTube video
2. List indexed videos
3. Ask a question
4. Search topics
...
```

**That's it!** Start by adding a YouTube video URL.

---

## Option 2: Ollama (Fully Local)

**Best for:** Privacy-conscious users. Fully offline, no internet needed after setup.

### Step 1: Install Ollama

Download from https://ollama.ai/ and follow the installation guide for your OS.

### Step 2: Start Ollama

```bash
ollama serve
```

This will print: `Listening on 127.0.0.1:11434`

Leave this terminal running.

### Step 3: Download a Model (in a new terminal)

```bash
# Download Mistral (7B, fast & good quality)
ollama pull mistral

# Or try these alternatives:
# ollama pull neural-chat       # Optimized for conversations
# ollama pull llama2            # Meta's Llama
# ollama pull dolphin-mixtral   # Mixtral variant
```

First download takes a few minutes depending on your internet.

### Step 4: Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Fully local setup
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral

# All free and local:
EMBEDDING_PROVIDER=local
RERANKER_PROVIDER=cross_encoder
```

### Step 5: Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 6: Run

```bash
python main.py
```

**Pro tip:** On first run, downloading embeddings model will take 1-2 minutes. Subsequent runs are instant.

---

## Option 3: Offline Demo

**Best for:** Testing without any setup or API keys.

```bash
python scripts/offline_demo.py
```

This runs the full pipeline on pre-loaded transcripts using the hash embedder (deterministic, no ML models needed).

---

## What's Free?

### LLM (Language Model)

| Provider | Cost | Notes |
|----------|------|-------|
| **Groq** | FREE tier | 30 requests/min, perfect for single user |
| **Ollama** | FREE | Fully local, unlimited |
| OpenAI | Paid | Not recommended |
| Fake | FREE | Testing only, returns context snippets |

### Embeddings (Text-to-Vector)

| Provider | Cost | Notes |
|----------|------|-------|
| **Local (sentence-transformers)** | FREE | Fully local, runs on CPU |
| OpenAI | Paid | Not recommended |
| Hash | FREE | Deterministic, lower quality, testing only |

### Reranking (Re-scoring Retrieved Chunks)

| Provider | Cost | Notes |
|----------|------|-------|
| **Cross-Encoder (local)** | FREE | Runs on CPU, included |
| Identity | FREE | No reranking, just uses RRF order |
| Cohere | Paid | Not recommended |

### Vector Storage

| Provider | Cost | Notes |
|----------|------|-------|
| **ChromaDB** | FREE | Persistent local database |
| In-Memory | FREE | Loses data on restart |

### YouTube Transcripts

| Provider | Cost | Notes |
|----------|------|-------|
| **youtube-transcript-api** | FREE | Public API, no key needed |

---

## Cost Summary

| Setup | LLM | Embeddings | Reranking | **Total** |
|-------|-----|-----------|-----------|----------|
| Groq (Recommended) | $0 | $0 | $0 | **$0** ✅ |
| Ollama (Fully Local) | $0 | $0 | $0 | **$0** ✅ |
| Offline Demo | $0 | $0 | $0 | **$0** ✅ |
| OpenAI Full | $$$ | $$$$ | $$ | **$$$$** ❌ Avoid |

---

## Troubleshooting

### "GROQ_API_KEY is not set"

**Problem:** You haven't configured Groq.

**Solution:**
1. Get a free key from https://console.groq.com/
2. Add `GROQ_API_KEY=your_key` to `.env`

---

### "Cannot connect to Ollama"

**Problem:** Ollama server isn't running.

**Solution:**
```bash
# In a separate terminal:
ollama serve
```

Make sure you see: `Listening on 127.0.0.1:11434`

---

### "sentence-transformers is not installed"

**Problem:** Embeddings model wasn't installed.

**Solution:**
```bash
pip install sentence-transformers
```

Or reinstall all dependencies:
```bash
pip install -r requirements.txt
```

---

### "Model file not found" (Ollama)

**Problem:** You haven't downloaded a model yet.

**Solution:**
```bash
ollama pull mistral
```

Or any model from https://ollama.ai/library

---

### Slow embeddings generation on first run

**Expected behavior:** On first run with local embeddings, the model downloads (~400MB) and caches. This takes 1-2 minutes.

Subsequent runs are **instant** because the model is cached.

**Tip:** Run `python scripts/offline_demo.py` first to cache the model, then use normally.

---

### "EMBEDDING_DIM mismatch"

**Problem:** You changed embedding models but forgot to update `EMBEDDING_DIM` in `.env`.

**Solution:**
Different models have different dimensions:
- `all-MiniLM-L6-v2`: 384 (recommended)
- `all-mpnet-base-v2`: 768
- `all-distilroberta-v1`: 768

Update your `.env`:
```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
```

Then regenerate embeddings by deleting the vector store:
```bash
rm -rf chroma_db/
```

---

### "Running out of free tier limits"

**Groq API:**
- Free tier: 30 requests/min
- Generous for single-user testing
- Limits reset hourly

**Solution:** Use Ollama instead (fully local, unlimited).

---

## Performance Tips

### For Faster Responses

1. **Use Groq** instead of Ollama (API call is faster than local inference)
2. **Use smaller embeddings model:**
   ```env
   EMBEDDING_PROVIDER=local
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   EMBEDDING_DIM=384
   ```

### For Better Quality

1. **Use Ollama with larger model:**
   ```bash
   ollama pull neural-chat:13b
   ```
   Then in `.env`: `LLM_MODEL=neural-chat:13b`

2. **Use larger embeddings model:**
   ```env
   EMBEDDING_MODEL=all-mpnet-base-v2
   EMBEDDING_DIM=768
   ```

### For Lower Memory Usage

1. **Use Groq** (runs on their servers)
2. **Use smaller embeddings:**
   ```env
   EMBEDDING_PROVIDER=hash
   EMBEDDING_DIM=256
   ```

---

## Examples

### Fast & Free Setup (Recommended)
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxx
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
RERANKER_PROVIDER=cross_encoder
```

### Fully Offline Setup
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
RERANKER_PROVIDER=cross_encoder
```

### Testing/Demo Setup
```env
LLM_PROVIDER=fake
EMBEDDING_PROVIDER=hash
EMBEDDING_DIM=256
RERANKER_PROVIDER=identity
```

---

## FAQ

**Q: Do I need a credit card?**
A: No. Groq and all local options are completely free.

**Q: Can I use this offline?**
A: Yes, with Ollama option. Once you download the model, everything runs locally.

**Q: How long does setup take?**
A: Groq: 2 minutes. Ollama: 5-10 minutes (first model download).

**Q: Can I switch providers later?**
A: Yes! Just update `.env` and `.env` and run again. Your videos stay in the database.

**Q: Does it work on my Mac/Linux/Windows?**
A: Yes, all options work on all platforms.

**Q: What if I want to upgrade to paid APIs later?**
A: Just update `.env`. The code abstracts providers, so it's a one-line change.

---

## Next Steps

1. Choose your setup (Groq recommended)
2. Follow the steps above
3. Run `python main.py`
4. Add a YouTube video URL
5. Ask questions!

Happy learning! 🚀
