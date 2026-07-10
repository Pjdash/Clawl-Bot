from langchain_chroma import Chroma
from Backend.models import get_embedding_model


PERSIST_DIRECTORY = "./chroma_db"

def create_vector_db(chunks):

    print("Step 1")
    embedding_model = get_embedding_model()

    print("Step 2")

    texts = []
    metadatas = []

    for chunk in chunks:
        texts.append(chunk["text"])
        metadatas.append({
            "url": chunk["url"],
            "title": chunk["title"]
        })

    print("Step 3")

    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=PERSIST_DIRECTORY
    )

    print("Step 4")

    return vector_db


def load_vector_db():

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )