# from openai import OpenAI
# from dotenv import load_dotenv
# from chromadb import PersistentClient
# from litellm import completion
# from pydantic import BaseModel, Field
# from pathlib import Path
# from tenacity import retry, wait_exponential
# import gradio as gr

# load_dotenv(override=True)

# # =========================
# # CONFIG
# # =========================

# MODEL = "ollama/llama3.2"

# OLLAMA_BASE_URL = "http://localhost:11434"

# DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")

# collection_name = "docs"

# embedding_model = "nomic-embed-text"

# wait = wait_exponential(multiplier=1, min=2, max=10)

# RETRIEVAL_K = 3
# FINAL_K = 3

# # =========================
# # OPENAI CLIENT
# # =========================

# openai = OpenAI(
#     api_key="ollama",
#     base_url=f"{OLLAMA_BASE_URL}/v1"
# )

# # =========================
# # CHROMA
# # =========================

# chroma = PersistentClient(path=DB_NAME)

# collection = chroma.get_or_create_collection(collection_name)

# # =========================
# # PROMPT
# # =========================

# SYSTEM_PROMPT = """
# You are a knowledgeable, accurate assistant representing Insurellm.
# You are answering questions using the provided knowledge base content and RAG concepts.
# Always use the documents in the context if they contain the answer.
# Do not invent facts or make up answers not supported by the context.
# If the answer cannot be determined from the provided context, say exactly:
# "I don't know based on the provided knowledge."
# If the user asks multiple questions, answer each one clearly and separately.
# When possible, include relevant source names or file paths from the context.
# If you can, explain edge cases, caveats, or limitations based on the available information.
# Context:
# {context}
# """
# # =========================
# # MODELS
# # =========================

# class Result(BaseModel):
#     page_content: str
#     metadata: dict


# class RankOrder(BaseModel):
#     order: list[int] = Field(
#         description="The order of chunk relevance from most relevant to least relevant"
#     )

# # =========================
# # QUERY REWRITE
# # =========================

# @retry(wait=wait)
# def rewrite_query(question, history=[]):
#     message = f"""
# You are helping retrieve information from a RAG knowledge base.
# Conversation History:
# {history}
# User Question:
# {question}
# Rewrite the question into a short, specific search query that focuses on the core concepts,
# entities, and terms needed to find the relevant knowledge base documents.
# IMPORTANT:
# - Respond ONLY with the rewritten query
# - No explanations
# - Keep it concise and precise
# """
#     response = completion(
#         model=MODEL,
#         messages=[
#             {
#                 "role": "system",
#                 "content": message
#             }
#         ]
#     )
#     rewritten = response.choices[0].message.content.strip()
#     print(f"\nRewritten Query: {rewritten}\n")
#     return rewritten

# # =========================
# # VECTOR SEARCH
# # =========================

# def fetch_context_unranked(question):
#     print("\nCreating query embedding...\n")
#     query_embedding = openai.embeddings.create(
#         model=embedding_model,
#         input=[question]
#     ).data[0].embedding
#     print("Searching Chroma...\n")
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=RETRIEVAL_K
#     )
#     chunks = []
#     for result in zip(
#         results["documents"][0],
#         results["metadatas"][0]
#     ):
#         chunks.append(
#             Result(
#                 page_content=result[0],
#                 metadata=result[1]
#             )
#         )
#     return chunks

# # =========================
# # MERGE CHUNKS
# # =========================

# def merge_chunks(chunks, reranked):

#     merged = chunks[:]

#     existing = [chunk.page_content for chunk in chunks]

#     for chunk in reranked:

#         if chunk.page_content not in existing:
#             merged.append(chunk)

#     return merged

# # =========================
# # RERANK
# # =========================

# @retry(wait=wait)
# def rerank(question, chunks):
#     if len(chunks) <= 1:
#         return chunks
#     user_prompt = f"""
# Question:
# {question}
# Rank these chunks from MOST relevant to LEAST relevant for answering the question.
# Use only the content in the chunks. Do not rely on external knowledge.
# Return ONLY a JSON object like:
# {{"order":[1,2,3]}}
# Chunks:
# """
#     for index, chunk in enumerate(chunks):
#         user_prompt += f"""
# CHUNK {index+1}:
# {chunk.page_content}
# """

#     response = completion(
#         model=MODEL,
#         messages=[
#             {
#                 "role": "user",
#                 "content": user_prompt
#             }
#         ]
#     )
#     reply = response.choices[0].message.content.strip()
#     print("\nReranker Response:\n")
#     print(reply)

#     try:
#         order = RankOrder.model_validate_json(reply).order
#         return [chunks[i - 1] for i in order]

#     except Exception as e:
#         print("\nReranking failed, using original order.\n")
#         return chunks

# # =========================
# # FETCH FINAL CONTEXT
# # =========================

# def fetch_context(original_question):

#     rewritten_question = rewrite_query(original_question)

#     chunks1 = fetch_context_unranked(original_question)

#     chunks2 = fetch_context_unranked(rewritten_question)

#     chunks = merge_chunks(chunks1, chunks2)

#     reranked = rerank(original_question, chunks)

#     return reranked[:FINAL_K]

# # =========================
# # CREATE CHAT MESSAGES
# # =========================

# def make_rag_messages(question, history, chunks):
#     context = "\n\n".join([
#         f"Source: {chunk.metadata['source']}\n{chunk.page_content}"
#         for chunk in chunks
#     ])
#     system_prompt = SYSTEM_PROMPT.format(
#         context=context
#     )
#     return (
#         [
#             {
#                 "role": "system",
#                 "content": system_prompt
#             }
#         ]
#         + history
#         + [
#             {
#                 "role": "user",
#                 "content": question
#             }
#         ]
#     )

# # =========================
# # ANSWER QUESTION
# # =========================

# @retry(wait=wait)
# def answer_question(question: str, history=[]):

#     chunks = fetch_context(question)

#     print("\nRetrieved Chunks:\n")

#     for i, chunk in enumerate(chunks):

#         print(f"\n--- Chunk {i+1} ---\n")

#         print(chunk.page_content[:400])

#     messages = make_rag_messages(
#         question,
#         history,
#         chunks
#     )

#     print("\nGenerating answer...\n")

#     response = completion(
#         model=MODEL,
#         messages=messages
#     )

#     return response.choices[0].message.content, chunks

# # =========================
# # MAIN LOOP
# # =========================

# if __name__ == "__main__":

#     print("\nRAG Chat Ready ✅\n")
#     history = []
#     while True:

#         question = input("\nAsk Question: ")

#         if question.lower() in ["exit", "quit"]:
#             break

#         answer, chunks = answer_question(
#             question,
#             history
#         )

#         print("\nAI Answer:\n")

#         print(answer)

#         history.append({
#             "role": "user",
#             "content": question
#         })

#         history.append({
#             "role": "assistant",
#             "content": answer
#         })






from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential
import gradio as gr

load_dotenv(override=True)

# =========================
# CONFIG
# =========================

MODEL = "llama3.2"  # no "ollama/" prefix — using openai client directly

OLLAMA_BASE_URL = "http://localhost:11434"

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")

collection_name = "docs"

embedding_model = "nomic-embed-text"

wait = wait_exponential(multiplier=1, min=2, max=10)

RETRIEVAL_K = 3
FINAL_K = 3

# =========================
# OPENAI CLIENT
# (used for both embeddings AND inference — no litellm overhead)
# =========================

openai = OpenAI(
    api_key="ollama",
    base_url=f"{OLLAMA_BASE_URL}/v1"
)

# =========================
# CHROMA
# =========================

chroma = PersistentClient(path=DB_NAME)

collection = chroma.get_or_create_collection(collection_name)

# =========================
# PROMPT
# =========================

SYSTEM_PROMPT = """
You are a knowledgeable, accurate assistant representing Insurellm.
You are answering questions using the provided knowledge base content and RAG concepts.
Always use the documents in the context if they contain the answer.
Do not invent facts or make up answers not supported by the context.
If the answer cannot be determined from the provided context, say exactly:
"I don't know based on the provided knowledge."
If the user asks multiple questions, answer each one clearly and separately.
When possible, include relevant source names or file paths from the context.
If you can, explain edge cases, caveats, or limitations based on the available information.
Context:
{context}
"""

# =========================
# MODELS
# =========================

class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of chunk relevance from most relevant to least relevant"
    )

# =========================
# HISTORY HELPER
# Converts list of message dicts to readable plain text for prompt injection.
# Prevents LLM from looping when raw dicts are passed as context.
# =========================

def history_to_text(history: list[dict]) -> str:
    text = ""
    for msg in history[-4:]:  # only last 2 turns (user + assistant x2)
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "")
        text += f"{role}: {content}\n"
    return text.strip()

# =========================
# QUERY REWRITE
# =========================

@retry(wait=wait)
def rewrite_query(question: str, history: list[dict] = []) -> str:
    history_text = history_to_text(history)

    prompt = f"""You are helping retrieve information from a RAG knowledge base.
Conversation History:
{history_text}

User Question:
{question}

Rewrite the question into a short, specific search query that focuses on the core concepts,
entities, and terms needed to find the relevant knowledge base documents.
IMPORTANT:
- Respond ONLY with the rewritten query
- No explanations
- Keep it concise and precise"""

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    rewritten = response.choices[0].message.content.strip()
    print(f"\nRewritten Query: {rewritten}\n")
    return rewritten

# =========================
# VECTOR SEARCH
# =========================

def fetch_context_unranked(question: str) -> list[Result]:
    print("\nCreating query embedding...\n")
    query_embedding = openai.embeddings.create(
        model=embedding_model,
        input=[question]
    ).data[0].embedding

    print("Searching Chroma...\n")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RETRIEVAL_K
    )

    chunks = []
    for result in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        chunks.append(
            Result(
                page_content=result[0],
                metadata=result[1]
            )
        )
    return chunks

# =========================
# MERGE CHUNKS
# =========================

def merge_chunks(chunks: list[Result], reranked: list[Result]) -> list[Result]:
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

# =========================
# RERANK
# =========================

@retry(wait=wait)
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    if len(chunks) <= 1:
        return chunks

    user_prompt = f"""Question:
{question}

Rank these chunks from MOST relevant to LEAST relevant for answering the question.
Use only the content in the chunks. Do not rely on external knowledge.
Return ONLY a JSON object like:
{{"order":[1,2,3]}}

Chunks:
"""
    for index, chunk in enumerate(chunks):
        user_prompt += f"\nCHUNK {index + 1}:\n{chunk.page_content}\n"

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    reply = response.choices[0].message.content.strip()
    print("\nReranker Response:\n")
    print(reply)

    try:
        order = RankOrder.model_validate_json(reply).order
        return [chunks[i - 1] for i in order]
    except Exception:
        print("\nReranking failed, using original order.\n")
        return chunks

# =========================
# FETCH FINAL CONTEXT
# =========================

def fetch_context(original_question: str, history: list[dict] = []) -> list[Result]:
    rewritten_question = rewrite_query(original_question, history)

    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)

    chunks = merge_chunks(chunks1, chunks2)

    reranked = rerank(original_question, chunks)

    return reranked[:FINAL_K]

# =========================
# CREATE CHAT MESSAGES
# =========================

def make_rag_messages(question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
    context = "\n\n".join([
        f"Source: {chunk.metadata['source']}\n{chunk.page_content}"
        for chunk in chunks
    ])
    system_prompt = SYSTEM_PROMPT.format(context=context)

    return (
        [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        + history
        + [
            {
                "role": "user",
                "content": question
            }
        ]
    )

# =========================
# ANSWER QUESTION
# (shared by both CLI and Gradio)
# =========================

@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Result]]:
    chunks = fetch_context(question, history)

    print("\nRetrieved Chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---\n")
        print(chunk.page_content[:400])

    messages = make_rag_messages(question, history, chunks)

    print("\nGenerating answer...\n")

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    return response.choices[0].message.content, chunks

# =========================
# CLI MAIN LOOP
# =========================

def run_cli():
    print("\nRAG Chat Ready ✅\n")
    history = []
    while True:
        question = input("\nAsk Question: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer, chunks = answer_question(question, history)

        print("\nAI Answer:\n")
        print(answer)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

# =========================
# GRADIO HELPERS
# =========================

def format_context(chunks: list[Result]) -> str:
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in chunks:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


# =========================
# GRADIO UI
# =========================

def run_gradio():
    def handle_message(
        message: str,
        history: list[dict],
        context_display: str
    ) -> tuple[str, list[dict], str]:
        """
        Single handler for message submit.
        Takes the textbox message + existing history,
        runs the full RAG pipeline once, returns:
          - cleared textbox
          - updated history with user + assistant messages
          - updated context panel
        """
        if not message.strip():
            return "", history, context_display

        try:
            # history here is all prior completed turns
            answer, chunks = answer_question(message, history)

            updated_history = history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": answer},
            ]
            return "", updated_history, format_context(chunks)

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\nMake sure Ollama is running at {OLLAMA_BASE_URL}"
            updated_history = history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": error_msg},
            ]
            return "", updated_history, f"<span style='color: red;'>{error_msg}</span>"

    try:
        with gr.Blocks(title="Insurellm Expert Assistant") as ui:
            gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

            with gr.Row():
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot()
                    message = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about Insurellm...",
                        show_label=False,
                    )

                with gr.Column(scale=1):
                    context_markdown = gr.Markdown(
                        label="📚 Retrieved Context",
                        value="*Retrieved context will appear here*",
                        container=True,
                        height=600,
                    )

            # Single submit handler — no .then() chain so it fires exactly once
            message.submit(
                handle_message,
                inputs=[message, chatbot, context_markdown],
                outputs=[message, chatbot, context_markdown]
            )

        print(f"\n✅ Gradio UI starting at http://127.0.0.1:7860")
        print(f"📍 Ollama URL: {OLLAMA_BASE_URL}")
        print(f"📚 Database: {DB_NAME}\n")
        ui.launch(inbrowser=True)

    except Exception as e:
        print(f"\n❌ Failed to start Gradio UI: {e}")
        import traceback
        traceback.print_exc()

# =========================
# ENTRYPOINT
# Switch between CLI and Gradio by changing USE_GRADIO
# =========================

USE_GRADIO = True  # Set to True to launch Gradio UI

if __name__ == "__main__":
    if USE_GRADIO:
        run_gradio()
    else:
        run_cli()