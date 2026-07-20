from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
import getpass
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document


docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embeddings = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
  documents=docs, 
  embedding=embeddings,
  persist_directory="chroma_db"
  )

result=vector_store.similarity_search("What is used for data analysis ?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
    

retreiver= vector_store.as_retriever()
docs= retreiver.invoke("Explain Deep Learning")

for d in docs:
    print(d.page_content)
    


