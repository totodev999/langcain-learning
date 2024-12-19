from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter

def file_filter(file_path:str)->bool:
    return file_path.endswith(".yml")

loader = GitLoader(
    clone_url="https://github.com/totodev999/simple-lambda-template",
    repo_path="./simple-lambda-template",
    branch="main",
    file_filter=file_filter,
)

raw_docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=20,chunk_overlap=0)
splitted_text = text_splitter.split_documents(raw_docs)


print(len(splitted_text))
