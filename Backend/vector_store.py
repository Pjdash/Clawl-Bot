from Backend.chromaDB import load_vector_db


def retrieve_documents(query, k=3):

    vector_db = load_vector_db()

    retriever = vector_db.as_retriever(
        search_kwargs={"k": k}
    )

    documents = retriever.invoke(query)

    return documents