Sure, here's a clean and informative `README.md` for your project:

---

# Grafiti + Gemini Adapter

A test project integrating [Graphiti Core](https://pypi.org/project/graphiti-core/) with **Google Generative AI (Gemini)** for semantic graph building, document embedding, relevance ranking, and LLM-backed processing.

---

## Overview

This project demonstrates how to plug Google’s Gemini models into the Graphiti framework via custom adapters:

- **LLM Client** – For structured JSON generation using Gemini.
- **Embedder** – Uses Gemini’s embedding models (`text-embedding-004` by default).
- **Cross Encoder** – Reranks passages using Gemini by classifying relevance.

---

## Features

- Gemini support via `google-generativeai`
- Graph-based semantic indexing and storage
- Structured prompt-response logic using Pydantic models
- Configurable LLM, embedder, and reranker clients
- Neo4j-backed knowledge graph
- Pre-commit checks and linters (Ruff, Prettier, etc.)

---

## Requirements

- Python 3.11+
- Google API Key for Gemini (`GOOGLE_API_KEY` in environment)
- Local Neo4j instance (default: `bolt://localhost:7687`, user: `neo4j`, password: `password`)

---

## Project Structure

```
grafiti-test/
├── google_adapters/         # Gemini-based implementations
│   ├── llm_client.py        # JSON-based LLM client
│   ├── embedder.py          # Embedding client
│   └── cross_encoder_client.py  # Relevance scorer
├── main.py                  # Sample usage and episode ingestion
├── .pre-commit-config.yaml  # Code quality hooks
├── .gitignore
├── pyproject.toml           # Build and tool configuration
```

---

## Usage

```bash
# Install dependencies
pip install -e .

# Set your Google API Key
export GOOGLE_API_KEY=your_key_here

# Start Neo4j locally (if not already running)

# Run the demo
python main.py
```

---

## Pre-commit Hooks

Install and activate:

```bash
pre-commit install
pre-commit run --all-files
```

Includes checks for:

- Ruff linting
- JSON/YAML/XML/TOML validation
- Debug statements, trailing whitespace, etc.
- GitGuardian secret scanning
- Prettier formatting

---

## Notes

- Embeddings default to `text-embedding-004` (768-dim).

---
