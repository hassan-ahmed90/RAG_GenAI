from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(r"Document Loader\deeplearning.pdf")
docs = loader.load()




from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

templete=ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI that summarize the Test"),
        ("human", "{data}")
        
    ]
)

prompt=templete.format_messages(data=docs[0].page_content)

model=ChatMistralAI(
    model="mistral-small-2506"
)

result= model.invoke(prompt)
print(result.content)