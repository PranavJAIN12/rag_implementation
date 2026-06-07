# Insurellm RAG (Retrieval-Augmented Generation) Assistant

A modern RAG chatbot built with **Ollama**, **ChromaDB**, and **Gradio** for intelligent knowledge base retrieval and question answering.

## 🚀 Features

- **Retrieval-Augmented Generation (RAG)** - Answers questions using your knowledge base
- **Query Rewriting** - Improves search accuracy with intelligent query reformulation
- **Smart Reranking** - Ranks retrieved documents by relevance
- **Vector Database** - Uses ChromaDB with embeddings for fast semantic search
- **Gradio UI** - Beautiful web interface for easy interaction
- **Local LLM** - Runs entirely on your machine with Ollama

## 📋 Prerequisites

- **Python 3.9+**
- **Ollama** running locally (http://localhost:11434)
- **Models**: `llama3.2` and `nomic-embed-text`

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd RAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment**
   ```bash
   cp .env.example .env
   # Edit .env if needed (optional - defaults work with local Ollama)
   ```

5. **Ingest knowledge base**
   ```bash
   python main.py --ingest
   # or directly
   python pro_implementation/ingest.py
   ```

## 🎯 Quick Start

**Start the Gradio UI:**
```bash
python main.py
```

The chatbot will open at `http://localhost:7860`

## 📁 Project Structure

```
RAG/
├── main.py                    # Main launcher
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
│
├── pro_implementation/
│   ├── answer.py              # RAG chat engine with Gradio UI
│   ├── ingest.py              # Knowledge base ingestion pipeline
│   └── app.py                 # Additional app logic
│
├── knowledge_base/
│   └── data/
│       ├── space_exploration.md
│       └── lost_library.md
│
└── preprocessed_db/           # Generated (ignored by git)
    └── chroma.sqlite3
```

## 🔄 How It Works

### Ingestion Pipeline (`ingest.py`)
1. Reads markdown files from `knowledge_base/data/`
2. Splits documents into chunks (500 tokens, 100 overlap)
3. Creates embeddings using `nomic-embed-text`
4. Stores vectors in ChromaDB

### RAG Pipeline (`answer.py`)
1. **Query Rewriting** - Reformulates user question for better retrieval
2. **Vector Search** - Retrieves top 3 chunks using semantic similarity
3. **Reranking** - Ranks retrieved chunks by relevance
4. **Generation** - Uses LLM to answer based on context

## 🛠 Configuration

Edit `pro_implementation/answer.py`:
```python
MODEL = "llama3.2"                  # LLM model
OLLAMA_BASE_URL = "http://localhost:11434"
RETRIEVAL_K = 3                     # Chunks to retrieve
FINAL_K = 3                         # Top chunks to use
embedding_model = "nomic-embed-text"
```

## 📝 Adding Knowledge Base Documents

1. Create `.md` files in `knowledge_base/data/`
2. Run ingestion:
   ```bash
   python pro_implementation/ingest.py
   ```
3. Restart the chat to load new documents

## 🐛 Troubleshooting

**"Connection refused" error**
- Make sure Ollama is running: `ollama serve`

**Missing models**
- Pull required models:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

**Slow responses**
- Reduce `RETRIEVAL_K` and `FINAL_K` in `answer.py`
- Lower chunk size in `ingest.py`

## 📦 Requirements

See `requirements.txt` for full dependencies:
- `openai` - For API integration with Ollama
- `chromadb` - Vector database
- `litellm` - LLM interface
- `gradio` - Web UI
- `langchain` - Text processing
- `pydantic` - Data validation
- `python-dotenv` - Environment management

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to:
- Add more knowledge base documents
- Improve prompts in `answer.py`
- Optimize RAG pipeline parameters
