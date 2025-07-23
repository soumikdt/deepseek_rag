# deepseek_rag


This is a minimal Retrieval-Augmented Generation (RAG) pipeline using:

- `LangChain` for document loading and retrieval
- `HuggingFaceEmbeddings` for local text embedding
- `ChromaDB` as vector store
- `DeepSeek` API for generating responses

## 📁 Structure

- `docs/sample.txt`: Source document
- `main.py` or notebook: Loads, embeds, retrieves, and queries
- `.env`: Contains `DEESEEK_API_KEY` (excluded via `.gitignore`)

## 🚀 Usage

1. Install requirements:
   ```bash
   pip install langchain langchain-community chromadb sentence-transformers openai python-dotenv
   ```
