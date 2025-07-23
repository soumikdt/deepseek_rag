import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("Set DEEPSEEK_API_KEY in .env")

# Set up DeepSeek client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"

# Load document
loader = TextLoader("docs/sample.txt")
docs = loader.load()

# Chunk text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create vectorstore
embedding_model = OpenAIEmbeddings()  # You can use HuggingFace if OpenAI embeddings are not suitable
vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="chroma_store")

# Ask user query
query = input("Ask a question about the document: ")

# Retrieve similar chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.get_relevant_documents(query)
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Construct prompt
prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

# Call DeepSeek API
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1024
)

# Show answer
print("\n🧠 Answer:")
print(response.choices[0].message.content)
