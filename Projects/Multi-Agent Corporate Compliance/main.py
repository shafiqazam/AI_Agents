from langchain_community.document_loaders import PyPDFLoader

# 1. Initialize the loader with the path to your PDF
loader = PyPDFLoader("./Projects/Multi-Agent Corporate Compliance/src/GDPR.pdf")
pages = loader.load_and_split()

# 3. View the content of the first page
print(len(pages))


