# DeepSeek RAG

Minimal RAG pipeline using LangChain, ChromaDB, MiniLM embeddings, and DeepSeek API.

## Repo Structure

deepseek_rag/
├── docs/sample.txt
├── DeepSeek_RAG_Demo.ipynb
├── .env             # DEESEEK_API_KEY (gitignored)
└── README.md

## Install
```bash
pip install langchain langchain-community chromadb sentence-transformers openai python-dotenv
# optional for SpaCy splitting
pip install spacy && python -m spacy download en_core_web_sm
```

Usage

jupyter lab DeepSeek_RAG_Demo.ipynb

	1.	Enter your question.
	2.	See retrieved chunks and DeepSeek answer.

Config
	•	Adjust chunk_size, chunk_overlap, and k in the notebook.
	•	Toggle between RecursiveCharacterTextSplitter and SpacyTextSplitter.

⸻



