from langchain_groq import ChatGroq

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.chat_history import InMemoryChatMessageHistory

from Backend.vector_store import get_retriever


# -----------------------------
# Chat History Store
# -----------------------------

store = {}


def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# -----------------------------
# LLM
# -----------------------------

_llm = None


def get_llm(api_key):

    global _llm

    if _llm is None:

        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=api_key,
            temperature=0.2
        )

    return _llm


# -----------------------------
# Build Conversational RAG Chain
# -----------------------------

def build_chain(api_key, top_k=5):

    llm = get_llm(api_key)

    retriever = get_retriever(top_k)

    # -----------------------------
    # Reformulate Follow-up Question
    # -----------------------------

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Given the chat history and the latest user question,
rewrite the question so it is self-contained.

Do NOT answer the question.

Only rewrite it if necessary.
"""
            ),

            MessagesPlaceholder("chat_history"),

            (
                "human",
                "{input}"
            )
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # -----------------------------
    # QA Prompt
    # -----------------------------

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Retrieval-Augmented Generation assistant.

Rules:

1. Use ONLY the retrieved context.

2. Do NOT use outside knowledge.

3. If the answer is unavailable, reply exactly:

"I cannot find the answer in the crawled website content."

4. Mention source URLs whenever relevant.

Context:

{context}
"""
            ),

            MessagesPlaceholder("chat_history"),

            (
                "human",
                "{input}"
            )
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    conversational_rag = RunnableWithMessageHistory(

        rag_chain,

        get_session_history,

        input_messages_key="input",

        history_messages_key="chat_history",

        output_messages_key="answer"

    )

    return conversational_rag


# -----------------------------
# Generate Answer
# -----------------------------

def generate_answer(
    query,
    api_key,
    session_id,
    top_k=5
):

    chain = build_chain(api_key, top_k)

    result = chain.invoke(

        {
            "input": query
        },

        config={
            "configurable": {
                "session_id": session_id
            }
        }

    )

    return result