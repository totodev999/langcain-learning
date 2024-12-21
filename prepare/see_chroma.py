from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import json

vector_store = Chroma(
    collection_name="japanese_companies",
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

docs = vector_store.get()
documents: list[str] = docs["documents"]

target_docs = []
for document in documents:
    if "震災" in document:
        target_docs.append(document)

for doc in target_docs:
    print("---------------Doc-----------------")
    print(doc)


# vector_store.delete(doc.get("ids"))

# openai_key = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = openai_key

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# vector_store = Chroma(
#     collection_name="japanese_companies",
#     persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
#     embedding_function=embeddings,
# )

# docs = vector_store.similarity_search(query="震災")
# for doc in docs:
#     print(doc)
#     print(doc.metadata)
