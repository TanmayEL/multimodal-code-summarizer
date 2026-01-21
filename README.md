## Multimodal Code Review Summarizer (Work‑in‑Progress)

This is a side project where I am slowly building a system that can:

- Take a **pull request** (PR) – code diff + PR text + comments  
- Look at both the **text** and a **visual representation** of the diff  
- Optionally pull in **extra context from the repo** (RAG)  
- And spit out a **short, human‑style summary** of what changed and why

It iss intentionally not polished “production SaaS”. It is  a realistic research / portfolio project that shows how I think, experiment, and structure code over time.

---

## What’s Done So Far (High‑Level)

- **Phase 1 – Data pipeline (done)**
  - Clean and process `git diff` text
  - Turn diffs into simple **images** (color bars for added/removed lines)
  - Combine PR context (title, description, comments)
  - Save everything into a `CodeReviewDataset` that PyTorch can use

- **Phase 2 – Multimodal model (in progress)**
  - Vision Transformer (**ViT**) for diff images
  - CodeBERT‑style encoder for diff text + PR context
  - Fusion layer to mix image + text features
  - Summary head that produces logits (first step toward text summaries)

- **RAG + LLM plumbing – Push 1 (done, mock mode)**
  - Simple repo index over code files (Python) using local “fake” embeddings
  - Retriever that grabs related code snippets for a PR
  - LLM wrapper (currently mocked) + prompt builder for PR summaries
  - CLI to run the pipeline end‑to‑end without touching any paid API

The idea is: start with a concrete data pipeline + model, then layer in retrieval + LLMs, then later expose everything as an API + UI.

---


## Phase 1 – Data Pipeline

### What problem this solves

Models wwant clean, consistent tensors, not random JSON or messy diffs.  
Phase 1 is about turning “real‑world PR data” into something a model can safely consume.

### Main pieces

- **`src/data/processors.py`**
  - **`CodeProcessor`**
    - strips git metadata (`diff --git`, `index`, file headers).
    - naive whitespace tokenization (good enough for now).
  - **`DiffImageProcessor`**
    - builds a simple RGB image where:
      - `+` lines are green, `-` lines are red, `@` lines are blue, others white.
      - Each line is a horizontal bar, some text is drawn on top for context.
  - **`ContextProcessor`**
    - merges PR title, description, and comments into one string and truncates it to a safe length.

- **`src/data/dataset.py`**
  - `CodeReviewDataset`:
    - On init, reads `train.json` or `val.json` from `data/processed/`.
    - Each item returns a dict with:
      - `diff_image`: tensor `[C, H, W]` made from the diff.
      - `diff_text`: cleaned diff string.
      - `context`: combined PR context string.
      - `summary`: target summary text (what we eventually want the model to predict).

- **`scripts/prepare_data.py`**
  - Reads raw `.json` files from `data/raw/` (each entry has `diff`, `title`, `description`, `comments`, `summary`).
  - Cleans + processes them using the processors above.
  - Splits into train/val using `train_test_split`.
  - Writes `data/processed/train.json` and `data/processed/val.json`.


## Phase 2 – Multimodal Model (Still Being Built)

### Intuition

The model should look at **both**:

- how the diff looks (rough patterns of additions/deletions), and  
- what the text says (actual code + descriptions),

then produce a representation we can decode into a summary.

### Components

- **`src/models/vision_transformer.py`**
  - Implements a mini **Vision Transformer (ViT)** for diff images:
    - image → patches → embeddings.
    - standard transformer blocks.
    - stacks blocks and returns a CLS token vector per image.

- **`src/models/code_bert.py`**
  - A “BERT‑ish” encoder for **diff text** and **PR context**:
    - token, position, and segment embeddings.
    - self‑attention over tokens.
    - runs both diff text and context through shared blocks and returns one vector for each (mean pooling + small MLP).

- **`src/models/fusion.py`**
  - **Multimodal fusion**:
    - allows image features to attend to text features and vice versa.
    - combines image, diff‑text, and context vectors into one fused representation.

- **`src/models/architecture.py`**
  - merges everything together

## RAG + LLM Plumbing (Mock Mode)

This part is about connecting PRs to **extra repo context** and then to an **LLM‑style summarizer**, even if the LLM is currently mocked.

- Indexing the repository
- Retrieving relevant code for a PR
- Building the LLM prompt
- LLM client (mock) + summarizer


## Tech Stack (What I’m Using)

- **Core ML / DL**: `torch`, custom transformers (no heavy HF wiring yet)
- **Image stuff**: `opencv-python`, `Pillow`
- **API / Web (planned)**: `fastapi`, `uvicorn`, `streamlit`
- **Data / Utils**: `numpy`, `pandas`, `scikit-learn`, `python-dotenv`
- **Dev tooling**: `pytest`, `black`, `flake8`, `mypy`, `mkdocs` (later)

---

## Roadmap (Rough)

- **Now (done / in progress)**
  - Data pipeline for PR diffs + context
  - Multimodal model skeleton (ViT + CodeBERT + fusion)
  - RAG + LLM mock integration (index → retrieve → prompt → mock LLM)

- **Next (future pushes)**
  - Swap `SimpleEmbedder` with real embeddings (OpenAI / HF)
  - Swap `LLMClient` mock with real GPT‑style model
  - Add **FastAPI** app:
    - `POST /summarize_pr` → returns summary JSON
  - Build **Streamlit** UI:
    - Paste PR / diff → see summary + context
  - Add caching by commit SHA + small evaluation scripts.

---
