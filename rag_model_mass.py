from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.callbacks.tracers import LangChainTracer
from langchain.schema.runnable import RunnableLambda

from constants import LLM_MODEL, EMBEDDING_MODEL, EXPLAIN_RESUME_MATCH_PROMPT

tracer = LangChainTracer()

llm = OllamaLLM(model = LLM_MODEL)
embedding_model = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

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

    prompt = EXPLAIN_RESUME_MATCH_PROMPT.format(
        full_resume_text = full_resume_text,
        jd_text = jd_text
    )

    response = llm.invoke(prompt, config={"callbacks": [tracer], "run_name": "Match Explanation LLM"})
    try:
        explanation = response.strip()
    except:
        explanation = "Could not generate explanation."

    return explanation