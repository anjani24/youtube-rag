# ✅ YouTube RAG - Free API Conversion Complete!

Your project has been successfully modified to work with **completely FREE APIs** requiring no credit card.

## 🚀 Quick Start (Choose One)

### Option 1: Groq API (Recommended - 2 minutes setup)
```bash
# 1. Get free key at: https://console.groq.com/
# 2. Edit .env and add: GROQ_API_KEY=your_key_here
# 3. Run:
python main.py
```
**Cost: $0** | Speed: ⚡⚡⚡ Fast | Setup: Easy

### Option 2: Ollama (Fully Local - 10 minutes setup)
```bash
# 1. Install from https://ollama.ai/
# 2. Download a model: ollama pull mistral
# 3. Start: ollama serve (in another terminal)
# 4. Update .env with ollama settings (see .env.example)
# 5. Run:
python main.py
```
**Cost: $0** | Speed: ⚡⚡ Medium | Setup: Medium | Privacy: ✅ Fully Local

### Option 3: Offline Demo (No setup)
```bash
python scripts/offline_demo.py
```
**Cost: $0** | Setup: None (for testing)

---

## 📊 What Changed?

| Component | Before | After | Cost |
|-----------|--------|-------|------|
| LLM | OpenAI (paid) | **Groq** (free) | **$0** ✅ |
| Embeddings | OpenAI (paid) | **Local** (free) | **$0** ✅ |
| Reranking | Identity | **Cross-Encoder** (free) | **$0** ✅ |
| Vector DB | Chroma | Chroma | $0 |
| YouTube API | youtube-transcript-api | youtube-transcript-api | $0 |

## 📁 Files Modified

1. **`app/config.py`** - Updated defaults to use free APIs
2. **`app/llm/provider.py`** - Added GroqLLM & OllamaLLM classes
3. **`app/embeddings/embedder.py`** - Updated docstrings & defaults
4. **`requirements.txt`** - Added free dependencies (groq, sentence-transformers)
5. **`.env.example`** - Complete rewrite with free setup instructions
6. **`README.md`** - Added free setup guide section
7. **`FREE_API_SETUP.md`** - NEW: Comprehensive free API setup guide (detailed instructions + troubleshooting)

## 🎯 Default Configuration (COMPLETELY FREE)

```env
# These are already set in .env.example
LLM_PROVIDER=groq              # Free Groq API
GROQ_API_KEY=your_key_here     # Get free at https://console.groq.com/
EMBEDDING_PROVIDER=local       # Local embeddings (free)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
RERANKER_PROVIDER=cross_encoder # Free local reranking
```

## 💡 Key Features

✅ **No Credit Card Needed** - Groq and Ollama are completely free
✅ **3 Setup Options** - Choose what works for you
✅ **Offline Mode Available** - Run Ollama locally with no internet
✅ **Better Quality** - Local cross-encoder reranking for better answers
✅ **Backwards Compatible** - Old OpenAI configs still work
✅ **Comprehensive Docs** - See FREE_API_SETUP.md for detailed guide

## 📚 Documentation

### For Quick Start
👉 Read: `README.md` - "Quick Start (FREE - No Credit Card Needed!)" section

### For Detailed Setup
👉 Read: `FREE_API_SETUP.md` - Complete guide with:
- Step-by-step setup for Groq
- Step-by-step setup for Ollama
- Troubleshooting
- Performance tips
- Cost comparison

### For Environment Variables
👉 Read: `.env.example` - Well-commented configuration template

## 🔧 Installation & Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies (includes free APIs)
pip install -r requirements.txt

# 3. Setup configuration
cp .env.example .env

# 4. Choose your setup:
# - Option A: Get Groq key from https://console.groq.com/
# - Option B: Install Ollama and download a model
# - Option C: Just run offline demo

# 5. Run
python main.py
```

## ⚡ LLM Provider Comparison

| Provider | Cost | Speed | Privacy | Offline | Setup Time |
|----------|------|-------|---------|---------|------------|
| **Groq** ⭐ | FREE (free tier) | Very Fast | 🔴 Cloud | No | 2 min |
| **Ollama** ⭐ | FREE (local) | Medium | ✅ Local | Yes | 10 min |
| **Fake** | FREE (local) | Instant | ✅ Local | Yes | 0 min |
| OpenAI | $$ Paid | Fast | 🔴 Cloud | No | 5 min |

**Groq recommended** - Best balance of speed, ease, and free tier generosity.

## 📦 What Gets Installed

All dependencies are **free and open-source**:

```
FREE (included in requirements.txt):
✅ youtube-transcript-api    - YouTube captions
✅ chromadb                  - Vector database
✅ rank-bm25                 - Keyword search
✅ sentence-transformers     - Local embeddings + reranking
✅ groq                      - Free LLM API
✅ python-dotenv             - Configuration
✅ requests, rich, numpy    - Utilities

OPTIONAL (commented out):
❌ openai                    - PAID, not needed with defaults
❌ cohere                    - PAID, not needed with defaults
```

## 🎓 Example Workflows

### Workflow 1: Fastest Setup (Recommended)
```bash
# Get free Groq key (2 minutes)
# Add to .env: GROQ_API_KEY=your_key
# Run
python main.py
```
✅ Fast | ✅ Free | ✅ Easy

### Workflow 2: Maximum Privacy (Fully Local)
```bash
# Install Ollama, pull mistral
# Update .env with ollama config
# Run
python main.py
```
✅ Free | ✅ Offline | ✅ Private | ⚠️ Slower

### Workflow 3: Testing/Demo
```bash
python scripts/offline_demo.py
```
✅ Free | ✅ No setup | ⚠️ Testing only

## ❓ Common Questions

**Q: Do I really need no credit card?**
A: Yes! Groq is completely free, and Ollama runs locally.

**Q: How long does setup take?**
A: Groq: ~2 minutes | Ollama: ~10 minutes | Demo: instant

**Q: Will my videos/data be deleted if I switch providers?**
A: No, your ChromaDB stays in `chroma_db/` folder and works with all providers.

**Q: Can I use OpenAI if I want?**
A: Yes! Just update `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your_key
```
Everything else is the same.

**Q: Is the quality different?**
A: Groq and OpenAI are similar quality. Ollama is slightly lower quality but still good.

**Q: What about the embeddings quality?**
A: `all-MiniLM-L6-v2` is very good (384-dim). For better quality, use `all-mpnet-base-v2` (768-dim).

## 🛠️ Troubleshooting

If you hit any issues:

1. **"GROQ_API_KEY not set"** → Get key at https://console.groq.com/
2. **"Cannot connect to Ollama"** → Run `ollama serve` in another terminal
3. **"sentence-transformers not installed"** → Run `pip install -r requirements.txt`
4. **Embeddings slow on first run** → Normal, model downloads & caches (~400MB, 1-2 min)

For more details: See `FREE_API_SETUP.md` → Troubleshooting section

## 📞 Need Help?

All setup help is in: **`FREE_API_SETUP.md`**

This file has:
- Detailed step-by-step instructions
- Screenshots/commands to copy-paste
- Full troubleshooting section
- Performance tuning tips
- FAQ

## 🎉 You're All Set!

Your YouTube RAG is now:
- ✅ Completely FREE (no credit card)
- ✅ Easy to set up (2-10 minutes)
- ✅ Ready to use (just add a YouTube video URL)
- ✅ Production-ready (with better reranking)

**Next steps:**
1. Follow Quick Start above
2. Run `python main.py`
3. Add a YouTube video URL
4. Ask questions!

Happy learning! 🚀
