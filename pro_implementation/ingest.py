from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import json

# To run this file: C:\venv_rag\Scripts\python.exe "C:\Users\Pranav Jain\Desktop\Personal\AI_Course\personal_project\RAG\pro-implementation\ingest.py"

load_dotenv(override=True)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "gpt-oss:120b-cloud" 
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "nomic-embed-text"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge_base"
WORKERS = 1
wait = wait_exponential(multiplier=1, min=2, max=10)

openai = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

class Result(BaseModel):
    page_content: str
    metadata: dict
    
def fetch_documents():
    documents =[]
    if not KNOWLEDGE_BASE_PATH.exists():
        return documents

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    documents.append({
                        "type": doc_type,
                        "source": file.as_posix(),
                        "text": f.read()
                    })
            except Exception:
                continue
            print(f"Loaded {len(documents)} documents", end="\r")

    return documents

def process_document(document):
    print(f"\nProcessing: {document['source']}\n")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_texts = splitter.split_text(document["text"])
    results = []
    
    for i, text in enumerate(split_texts):
        page_content = f"""
Chunk {i+1}
    
{text}
"""
    
        metadata = {
            "source": document["source"],
            "type": document["type"]
        }
        
        results.append(
            Result(
                page_content=page_content,
                metadata=metadata
            )
        )
    print(f"Created {len(results)} chunks")
    
    return results

def create_chunks(documents):

    chunks = []

    with Pool(processes=WORKERS) as pool:

        for result in tqdm(
            pool.imap_unordered(process_document, documents),
            total=len(documents)
        ):
            chunks.extend(result)

    print(f"\nTotal chunks: {len(chunks)}\n")

    return chunks

def create_embeddings(chunks):
    
    print("\n Creating Embeddings...")
    
    chroma = PersistentClient(path=DB_NAME)
    
    existing = [c.name for c in chroma.list_collections()]
    
    if collection_name in existing:
        chroma.delete_collection(collection_name)
        
    texts = [chunk.page_content for chunk in chunks]
    
    emb_response = openai.embeddings.create(
        model = embedding_model,
        input = texts
    )
    
    vectors = [e.embedding for e in emb_response.data]
    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metas
    )

    print(f"\nVector DB created with {collection.count()} documents\n")


if __name__ == "__main__":
    print("\nStarting ingestion pipeline...\n")
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    
    print(f"\nIngestion Complete ")