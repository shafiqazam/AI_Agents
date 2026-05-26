import os 
import shutil
import chromadb

DB_PATH = "./chroma_db"
COLLECTION_NAME = "company_policy"

def initialize_database():

    if os.path.exists(DB_PATH):
        print(f"Removing existing database at {DB_PATH} for a clean slate.")
        shutil.rmtree(DB_PATH)

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    documents = [
        "Code of Conduct & Professional Ethics: Our company is committed to maintaining a workplace characterized by integrity, mutual respect, and professionalism. This policy outlines the behavioral standards expected of all team members, contractors, and representatives.",
        "Remote Work & Flexible Hours Policy: We recognize that flexibility can enhance productivity and support a healthy work-life balance. This policy defines the guidelines for remote work arrangements and core operational hours.",
        "Data Protection & Information Security: As a data-driven organization, protecting our proprietary information and customer data is paramount. This policy establishes the security protocols all employees must follow to safeguard digital assets.",
        "Annual Leave & Time-Off Policy: We believe that regular rest and time away from work are essential for sustained performance and personal well-being. This policy outlines how paid time off (PTO) is accrued and requested."
    ]

    docs_id = [f"doc_{i}" for i in range(len(documents))]

    print("Adding documents to ChromaDB collection...")
    collection.add(
        documents=documents,
        ids=docs_id
    )

    print(f"Database initialized with {len(documents)} documents in collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    initialize_database()