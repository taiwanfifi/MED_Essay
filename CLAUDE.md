# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project directory (國科會_RxLLama / National Science Council RxLLama) from Taipei Medical University focused on medical data engineering, clinical trial standards (BRIDG), and LLM applications in healthcare.

## Repository Structure

```
MDE_TMU/
├── 國科會_RxLLama/                    # Main LLM research project
│   ├── Medical-RAG-using-Bio-Mistral-7B-main/   # RAG system for medical QA
│   └── 關聯資料/                       # Drug-disease relationship analysis
│       ├── text2sql/                  # Natural language to SQL system
│       └── vanna/                     # Vanna-based Text2SQL implementation
├── BRIDGE/BRIDG/                      # CDISC BRIDG clinical trial standards
└── 參考/                              # Reference implementations and datasets
```

## Primary Subprojects

### 1. Medical-RAG (國科會_RxLLama/Medical-RAG-using-Bio-Mistral-7B-main/)

A Retrieval-Augmented Generation system for biomedical question-answering.

**Tech Stack:**
- LLM: BioMistral-7B (quantized GGUF via llama-cpp-python)
- Embeddings: PubMedBERT (NeuML/pubmedbert-base-embeddings)
- Vector DB: Qdrant (self-hosted via Docker)
- Framework: FastAPI + LangChain 0.1.8

**Setup:**
```bash
# Environment
conda create -n clean_env python=3.11 && conda activate clean_env
pip install -r requirements.txt
pip install langchain_community sentence-transformers langchain-huggingface
pip install "unstructured[pdf]" qdrant-client llama-cpp-python uvicorn langchain-qdrant

# Download model (place in project root)
# BioMistral-7B.Q4_K_M.gguf from https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF

# Start Qdrant
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant

# Ingest documents
python ingest.py

# Run app
python app.py  # Access at http://localhost:8000
```

**Key Configuration (hardcoded in app.py):**
- Model: `BioMistral-7B.Q4_K_M.gguf`
- Qdrant: `http://localhost:6333`
- Collection: `vector_db`
- Retrieval: top-3 documents (k=3)
- Chunk size: 700 tokens, overlap: 70

### 2. Text2SQL (國科會_RxLLama/關聯資料/text2sql/)

Converts natural language to SQL queries for medical data analysis using OpenAI GPT.

**Run:**
```bash
cd 國科會_RxLLama/關聯資料/text2sql
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
# Use text2sql_demo.ipynb or import from flow.py
```

**Components:**
- `data_processor.py` - CSV to SQLite import
- `sql_generator.py` - NL to SQL via GPT
- `query_executor.py` - Execute SQL
- `result_validator.py` - Validate results
- `flow.py` - Main orchestration

### 3. BRIDG Standards (BRIDGE/BRIDG/)

CDISC BRIDG 5.3.1 clinical trial data model documentation and mapping templates. Contains:
- Domain information models (EAP, XMI, PDF)
- Mapping spreadsheet templates
- Protocol templates and examples

## Common Commands

```bash
# Medical-RAG
cd 國科會_RxLLama/Medical-RAG-using-Bio-Mistral-7B-main
python ingest.py          # Index PDFs from data/ into Qdrant
python app.py             # Start FastAPI server on :8000
python test_embedding.py  # Test PubMedBERT embeddings

# Text2SQL
cd 國科會_RxLLama/關聯資料/text2sql/src
python data_processor.py  # Each module has __main__ for testing
python flow.py            # Run full pipeline

# Qdrant Dashboard
# http://localhost:6333/dashboard
```

## Architecture Notes

**Medical-RAG Flow:**
```
Query → PubMedBERT Embedding → Qdrant Similarity Search →
Top-3 Documents → Prompt Template → BioMistral-7B → Answer + Sources
```

**Text2SQL Flow:**
```
Natural Language Question → GPT (schema-aware SQL generation with CoT) →
SQL Execution → Result Validation → Iterative Refinement if needed
```

## Known Issues

- All configuration is hardcoded (model paths, URLs, hyperparameters)
- No error handling for missing documents in retrieval
- Synchronous LLM calls block the FastAPI server
- No authentication on web endpoints
