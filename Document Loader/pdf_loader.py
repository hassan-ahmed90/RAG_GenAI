from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"Document Loader\GRU.pdf")

docs = loader.load()
print(len(docs))