import os
import uvicorn

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv, set_key

from Backend.crawler import WebCrawler
from Backend.injest import process_documents
from Backend.generator import generate_answer
...
from Backend.chromaDB import (
    create_vector_db,
    load_vector_db
)

from Backend.vector_store import retrieve_documents
from Backend.bm25_store import create_bm25_index

load_dotenv()

app = FastAPI(
    title="RAG-Powered Website Chatbot API"
)

crawl_state = {
    "status": "idle",
    "current_url": "",
    "pages_crawled": 0,
    "queue_size": 0,
    "message": "",
    "total_chunks": 0
}
class SettingsRequest(BaseModel):
    groq_api_key: str


class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 30
    depth_limit: int = 2
    chunk_size: int = 600
    chunk_overlap: int = 100


class QueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 8


def get_api_key():

    key = os.environ.get("GROQ_API_KEY")

    if not key:
        raise HTTPException(
            status_code=400,
            detail="Groq API Key is not configured."
        )

    return key

def run_crawl_and_ingest_task(
    url,
    max_pages,
    depth_limit,
    chunk_size,
    chunk_overlap
):
    global crawl_state

    crawl_state["status"] = "crawling"
    crawl_state["message"] = "Initializing crawler..."

    crawler = WebCrawler(
        max_pages=max_pages,
        depth_limit=depth_limit
    )

    def on_progress(msg, count, q_size):

        crawl_state["current_url"] = msg
        crawl_state["pages_crawled"] = count
        crawl_state["queue_size"] = q_size

        crawl_state["message"] = (
            f"Scraping {count} pages..."
        )

    documents = crawler.crawl(
        url,
        progress_callback=on_progress
    )

    if not documents:

        crawl_state["status"] = "failed"

        crawl_state["message"] = (
            "No pages found."
        )

        return

    crawl_state["status"] = "indexing"

    crawl_state["message"] = (
        "Splitting pages..."
    )

    chunks = process_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not chunks:

        crawl_state["status"] = "failed"

        crawl_state["message"] = (
            "No chunks generated."
        )

        return

    try:

        crawl_state["message"] = (
            "Creating embeddings..."
        )

        print("Creating Chroma...")

        create_vector_db(chunks)

        print("Creating BM25...")

        create_bm25_index(chunks)

        print("Done.")

        crawl_state["status"] = "completed"

        crawl_state["message"] = (
            f"Indexed {len(chunks)} chunks."
        )

        crawl_state["total_chunks"] = len(chunks)

    except Exception as e:

        crawl_state["status"] = "failed"

        crawl_state["message"] = str(e)

@app.get("/api/settings")
def get_settings():
    key = os.environ.get("GROQ_API_KEY", "")

    has_key = len(key) > 0

    masked_key = (
        f"{key[:4]}...{key[-4:]}"
        if len(key) > 8
        else "****"
        if has_key
        else ""
    )

    return {
        "has_key": has_key,
        "masked_key": masked_key
    }


@app.post("/api/settings")
def save_settings(req: SettingsRequest):

    key = req.groq_api_key.strip()

    if not key:
        raise HTTPException(
            status_code=400,
            detail="API key cannot be empty."
        )

    env_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".env"
        )
    )

    set_key(env_path, "GROQ_API_KEY", key)

    os.environ["GROQ_API_KEY"] = key

    return {
        "status": "success",
        "message": "Groq API key saved."
    }

@app.post("/api/crawl")
def start_crawl(
    req: CrawlRequest,
    background_tasks: BackgroundTasks
):

    global crawl_state

    if (
        not req.url.startswith("http://")
        and
        not req.url.startswith("https://")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL."
        )

    if crawl_state["status"] in [
        "crawling",
        "indexing"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Crawler already running."
        )

    background_tasks.add_task(
        run_crawl_and_ingest_task,
        req.url,
        req.max_pages,
        req.depth_limit,
        req.chunk_size,
        req.chunk_overlap
    )

    crawl_state = {

        "status": "crawling",

        "current_url": req.url,

        "pages_crawled": 0,

        "queue_size": 0,

        "message": "Started crawling.",

        "total_chunks": 0
    }

    return {
        "status": "success"
    }

@app.get("/api/crawl/status")
def crawl_status():

    return crawl_state


@app.post("/api/query")
def run_query(req: QueryRequest):

    api_key = get_api_key()

    try:

        result = generate_answer(
            query=req.query,
            api_key=api_key,
            session_id=req.session_id,
            top_k=req.top_k
        )

        answer = result["answer"]

        documents = result["context"]

        sources = []

        seen = set()

        for doc in documents:

            url = doc.metadata.get("url")

            if url not in seen:

                seen.add(url)

                sources.append({

                    "url": url,

                    "title": doc.metadata.get(
                        "title",
                        "Untitled"
                    )

                })

        chunks = []

        for doc in documents:

            chunks.append({

                "url": doc.metadata.get("url"),

                "title": doc.metadata.get("title"),

                "text": doc.page_content

            })

        return {

            "answer": answer,

            "sources": sources,

            "chunks": chunks

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/sources")
def get_sources():

    return {

        "sources": [],

        "message":
        "Source listing isn't implemented for LangChain Chroma yet."

    }
import shutil


@app.post("/api/sources/clear")
def clear_sources():

    global crawl_state

    db_path = "./chroma_db"

    if os.path.exists(db_path):

        shutil.rmtree(db_path)

    crawl_state = {

        "status": "idle",

        "current_url": "",

        "pages_crawled": 0,

        "queue_size": 0,

        "message": "Database cleared.",

        "total_chunks": 0

    }

    return {

        "status": "success"

    }