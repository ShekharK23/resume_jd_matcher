from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableLambda

from langchain.callbacks.tracers import LangChainTracer
tracer = LangChainTracer()

llm = OllamaLLM(model="gemma3:1b")

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

def normalize_score(distance):
        similarity = 1 / (1 + distance)
        return round(similarity * 10, 2)

def calculate_resume_similarity(faiss_db, jd_text, top_n=5, tracer=None):
    """
    faiss_db: FAISS vector store containing all resumes
    jd_text: job description string
    top_n: number of top resumes to return
    tracer: optional LangChain tracer
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    embed_query = (
        RunnableLambda(lambda x: embedding_model.embed_query(x))
        .with_config(
            run_name="Embed Query",
            tags=["embedding", "query"],
            callbacks=[tracer] if tracer else None,
        )
    )

    query_vector = embed_query.invoke(jd_text)

    results = faiss_db.similarity_search_with_score_by_vector(query_vector, k=top_n)

    matches = []
    for doc, distance in results:  # FAISS returns (doc, distance)
        matches.append({
            "file_name": doc.metadata["file_name"],
            "score": normalize_score(distance),
            "content": doc.page_content
        })

    return matches


def explain_match(full_resume_text, jd_text):
    prompt = f"""
You are a helpful recruitment assistant.

You will be given:
1. A FULL RESUME
2. A JOB DESCRIPTION (JD)

Your task is to explain **why this candidate is a perfect fit for the job**.

Instructions:
- Give EXACTLY 5 bullet points.
- Focus on the strongest matches between the resume and JD.
- Mention relevant skills, tools, experiences, and achievements that align with the JD.
- Each point should start with: "- "

RESUME:
\"\"\"{full_resume_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

WHY PERFECT FIT:
"""

#     prompt = f"""
# You are a helpful recruitment assistant.

# You will be given:
# 1. A FULL RESUME
# 2. A JOB DESCRIPTION (JD)

# Your task is to explain **why this candidate is a perfect fit for the job**.

# STRICT FORMAT RULES:
# - Output EXACTLY 5 bullet points. Not more. Not less.
# - Each bullet point MUST start with "- " (dash + space).
# - Do not include an introduction or conclusion — only the 5 bullet points.
# - Each point should be a single sentence, concise and clear.
# - Focus ONLY on the strongest matches between the resume and JD (skills, tools, experience, achievements).
# - If you cannot find matches, still produce 5 points using the closest relevant details.

# RESUME:
# \"\"\"{full_resume_text}\"\"\"

# JOB DESCRIPTION:
# \"\"\"{jd_text}\"\"\"

# FINAL OUTPUT (exactly 5 bullet points):
# """

    response = llm.invoke(prompt, config={"callbacks": [tracer], "run_name": "Match Explanation LLM"})
    try:
        explanation = response.strip()
    except:
        explanation = "Could not generate explanation."

    return explanation