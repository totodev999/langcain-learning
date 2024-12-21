from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="japanese_companies",
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

doc = vector_store.get()
print(doc)

# vector_store.delete(doc.get("ids"))