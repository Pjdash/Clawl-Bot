from rank_bm25 import BM25Okapi
import pickle
import os

BM25_PATH = "./bm25_index.pkl"

bm25 = None
documents = None


def create_bm25_index(chunks):
    """
    chunks:
    [
        {
            "text": "...",
            "url": "...",
            "title": "..."
        }
    ]
    """

    global bm25
    global documents

    documents = chunks

    tokenized_docs = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_docs)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "documents": documents
            },
            f
        )


def load_bm25():

    global bm25
    global documents

    if bm25 is None:

        if not os.path.exists(BM25_PATH):
            return None

        with open(BM25_PATH, "rb") as f:

            data = pickle.load(f)

            bm25 = data["bm25"]
            documents = data["documents"]

    return bm25, documents


def bm25_search(query, k=5):

    result = load_bm25()

    if result is None:
        return []

    bm25, docs = result

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:k]