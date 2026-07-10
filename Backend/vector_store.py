from langchain_core.documents import Document

from Backend.chromaDB import load_vector_db
from Backend.bm25_store import bm25_search


def get_retriever(k=3):

    vector_db = load_vector_db()

    retriever = vector_db.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever


def retrieve_documents(query, k=3):

    # -----------------------------
    # Chroma Retrieval
    # -----------------------------

    retriever = get_retriever(k)

    chroma_docs = retriever.invoke(query)

    # -----------------------------
    # BM25 Retrieval
    # -----------------------------

    bm25_results = bm25_search(query, k)

    bm25_docs = []

    for chunk, score in bm25_results:

        bm25_docs.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "url": chunk["url"],
                    "title": chunk["title"]
                }
            )
        )

    # -----------------------------
    # Merge Results
    # -----------------------------

    merged_docs = []

    seen = set()

    # Add Chroma results first
    for doc in chroma_docs:

        key = (
            doc.page_content,
            doc.metadata.get("url", "")
        )

        if key not in seen:
            seen.add(key)
            merged_docs.append(doc)

    # Add BM25 results
    for doc in bm25_docs:

        key = (
            doc.page_content,
            doc.metadata.get("url", "")
        )

        if key not in seen:
            seen.add(key)
            merged_docs.append(doc)

    print(f"Chroma Retrieved : {len(chroma_docs)}")
    print(f"BM25 Retrieved   : {len(bm25_docs)}")
    print(f"Hybrid Retrieved : {len(merged_docs)}")

    return merged_docs