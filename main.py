from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()

loader = PyPDFLoader(r"Document Loader\deeplearning.pdf")
docs = loader.load()

splitter =RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap=200
)
chunks =splitter.split_documents(docs)
templete=ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI that summarize the Test"),
        ("human", "{data}")
        
    ]
)



model=ChatMistralAI(
    model="mistral-small-2506"
)
# prompt=templete.format_messages(data=docs)

# result= model.invoke(prompt)

# print(result.content)