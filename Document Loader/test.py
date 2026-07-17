from langchain_community.document_loaders import TextLoader

loader = TextLoader(r"Document Loader\notes.txt")

docs = loader.load()
print(docs)