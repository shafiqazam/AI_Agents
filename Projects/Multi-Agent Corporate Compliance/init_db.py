import pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def process_pdf_to_documents(file_path: str):
    """
    Loads a PDF and returns a list of LangChain Document objects.
    Each Document contains the page_content and metadata (page number).
    """
    # 1. Initialize the loader
    loader = PyPDFLoader(file_path)
    
    # 2. Load the data
    # By default, mode="page" will return a list where each item 
    # is a Document representing one page of the PDF.
    documents = loader.load()
    
    return documents

processed_documents = process_pdf_to_documents("./src/GDPR.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(processed_documents)

from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    collection_name="gdpr_collection",
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="./chroma_langchain_db",
)

print("Database seeded successfully!")