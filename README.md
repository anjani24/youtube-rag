# YouTube RAG

This is a small Python project for searching YouTube transcript content and answering questions from the video text.

It can:

- pull transcript text from a YouTube video
- split the transcript into chunks
- store chunks in a local vector database
- search relevant sections by meaning or keywords
- answer with citations that point back to timestamped transcript chunks

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run the app

```bash
python main.py
```

The menu lets you:

- add a video
- list indexed videos
- ask a question
- search topics
- summarize a video
- compare videos
- run evaluation

## Local config

Set your API keys in the `.env` file if you want to use Groq/OpenAI. If you do not want to use external services, the project also has local or offline options.

## Evaluate

```bash
python main.py evaluate
```

## Project layout

```text
youtube-rag/
├── app/
├── data/
├── scripts/
├── tests/
├── main.py
├── requirements.txt
├── README.md
├── .env.example
└── chroma_db/
```

This project is intentionally simple and local-first. The main idea is to keep transcript data searchable without building a website or backend around it.

A few practical notes:

- transcript quality depends on what YouTube provides
- captions can be noisy, so some cleanup is needed
- the local hash embedder is useful for offline testing, but stronger models are better for real work
- the app is easiest to understand if you read the code in the `app/` folder and keep the CLI as the main interface

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

If the test runner is not installed yet in your environment, install the project dependencies first and then run the command above.
