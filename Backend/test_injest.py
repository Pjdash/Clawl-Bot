from injest import process_documents

documents = [
    {
        "url": "https://example.com",
        "title": "Example Page",
        "text": (
            "Artificial Intelligence is transforming healthcare. " * 30
        )
    }
]

chunks = process_documents(
    documents,
    chunk_size=100,
    chunk_overlap=20
)

print("Total Chunks:", len(chunks))

for chunk in chunks:
    print("=" * 50)
    print("Chunk Index:", chunk["chunk_index"])
    print("Length:", len(chunk["text"]))
    print(chunk["text"])