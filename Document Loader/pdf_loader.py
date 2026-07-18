from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"Document Loader\GRU.pdf")

docs = loader.load()
print(len(docs))


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=10)

chunks= splitter.split_documents(docs)
print(len(chunks))
print(chunks[0].page_content)


# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader(r"Document Loader\GRU.pdf")

# docs = loader.load()
# print(len(docs))


# from langchain_text_splitters import TokenTextSplitter

# splitter = TokenTextSplitter(
#     chunk_size=1000, 
#     chunk_overlap=10)

# chunks= splitter.split_documents(docs)
# print(len(chunks))
# print(chunks[0].page_content)