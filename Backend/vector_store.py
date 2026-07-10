from Backend.chromaDB import load_vector_db


def get_retriever(k=3):

    vector_db = load_vector_db()

    retriever = vector_db.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever


def retrieve_documents(query, k=3):

    retriever = get_retriever(k)

    return retriever.invoke(query)