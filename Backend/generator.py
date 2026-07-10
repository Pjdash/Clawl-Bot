from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_llm = None
_current_key = None


def get_llm(api_key):
    global _llm, _current_key

    if _llm is None or api_key != _current_key:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=api_key,
            temperature=0.2
        )
        _current_key = api_key

    return _llm


def generate_answer(query, retrieved_docs, api_key):

    llm = get_llm(api_key)

    context = ""

    for idx, doc in enumerate(retrieved_docs):

        context += f"""
SOURCE {idx + 1}

URL:
{doc.metadata.get("url", "")}

TITLE:
{doc.metadata.get("title", "")}

CONTENT:
{doc.page_content}

====================================================
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Retrieval-Augmented Generation (RAG) assistant.

Rules:

1. Use ONLY the provided context.

2. Do NOT use outside knowledge.

3. If the answer is not found, reply exactly:

"I cannot find the answer in the crawled website content."

4. Be concise.

5. Mention source URLs whenever relevant.
"""
            ),
            (
                "human",
                """
Context:

{context}

Question:

{question}
"""
            )
        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    return answer