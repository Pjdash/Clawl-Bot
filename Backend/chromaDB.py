from langchain_chroma import Chroma
from Backend.models import get_embedding_model


PERSIST_DIRECTORY = "./chroma_db"


def create_vector_db(chunks):
    """
    chunks =
    [
        {
            "text": "...",
            "url": "...",
            "title": "..."
        }
    ]
    """

    embedding_model = get_embedding_model()

    texts = []
    metadatas = []

    for chunk in chunks:
        texts.append(chunk["text"])

        metadatas.append({
            "url": chunk["url"],
            "title": chunk["title"]
        })

    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_db


def load_vector_db():

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )