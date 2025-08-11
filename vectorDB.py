from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Store Chunks of resume into FIASS Vector DB
def store_embeddings(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_stores = {}

    for section, content in chunks.items():
        docs = [Document(page_content=content, metadata={"section": section})]
        faiss_db = FAISS.from_documents(docs, embedding_model)
        vector_stores[section] = faiss_db

    return vector_stores

# Store mass embedddings of resumes into FIASS Vector DB
def store_embeddings_mass(resumes):
    """
    resumes: list of dicts like:
    [
        {"file_name": "resume1.pdf", "content": "full resume text here"},
        {"file_name": "resume2.docx", "content": "full resume text here"}
    ]
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = None
    documents = []

    for resume in resumes:
        documents.append(
            Document(
                page_content=resume["content"],
                metadata={"file_name": resume["file_name"]}
            )
        )

    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store
