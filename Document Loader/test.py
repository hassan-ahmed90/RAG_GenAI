# 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=50,
    chunk_overlap=1
)

loader = TextLoader(r"Document Loader\notes.txt")
docs = loader.load()

chunks = splitter.split_documents(docs)

print(len(chunks))