from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.apple.com/newsroom/2026/03/apple-introduces-the-new-macbook-air-with-m5/")

docs = loader.load()
print(docs[0].page_content)