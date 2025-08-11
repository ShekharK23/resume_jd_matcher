
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.runnables import RunnableLambda
import re

from constants import LLM_MODEL, EMBEDDING_MODEL, RESUME_FEEDBACK_QUERY, SCORE_ALL_SECTIONS_PROMPT, SECTION_PATTERNS

from dotenv import load_dotenv
load_dotenv()

tracer = LangChainTracer()

llm = OllamaLLM(model = LLM_MODEL)
embedding_model = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

def chunk_by_section(text):

    header_regex = re.compile(
        r'(?P<header>(' + '|'.join(SECTION_PATTERNS.values()) + r'))\s*[:\n]', re.IGNORECASE
    )

    matches = list(header_regex.finditer(text))
    chunks = {}

    for i, match in enumerate(matches):
        header = match.group('header').strip().lower()
        start = match.end()

        section_key = None
        for key, pattern in SECTION_PATTERNS.items():
            if re.fullmatch(pattern, header, re.IGNORECASE):
                section_key = key
                break

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if section_key:
            section_text = text[start:end].strip()
            chunks[section_key] = section_text

    return chunks

def calculate_similarity(vector_stores, jd_text, tracer=None):

    # Trace only the embedding step
    embed_query = (
        RunnableLambda(lambda x: embedding_model.embed_query(x))
        .with_config(
            run_name="Embed Query",
            tags=["embedding", "query"],
            callbacks=[tracer] if tracer else None,
        )
    )

    query_vector = embed_query.invoke(jd_text)

    # Plain FAISS search (no tracing)
    matches = {}
    for section, faiss_db in vector_stores.items():
        docs = faiss_db.similarity_search_by_vector(query_vector, k=1)
        matches[section] = docs[0].page_content if docs else ""

    return matches


def get_scores_for_all_sections(sections_dict, jd_text):

    sections_text = "\n\n".join(
        f"[{section}]\n{content}" for section, content in sections_dict.items()
    )

    section_names_list = "\n".join(f"- {section}" for section in sections_dict.keys())
    
    prompt = SCORE_ALL_SECTIONS_PROMPT.format(
        section_names_list = section_names_list,
        sections_text = sections_text,
        jd_text = jd_text
    )

    response = llm.invoke(prompt, config={"callbacks": [tracer], "run_name": "Score LLM"})

    try:
        score_text = re.search(r"SCORES:\s*(.*)", response, re.DOTALL).group(1).strip()
        scores = {
            section.strip(): int(score.strip())
            for section, score in [
                line.strip().split(":", 1)
                for line in score_text.splitlines()
                if ':' in line
            ]
        }
    except Exception as e:
        scores = {}
        print("Parsing error:", e)
        print("LLM response:\n", response)

    return scores

def get_resume_level_feedback(full_resume_text, jd_text):

    prompt = RESUME_FEEDBACK_QUERY.format(
        full_resume_text=full_resume_text,
        jd_text=jd_text
    )

    response = llm.invoke(prompt, config={"callbacks": [tracer], "run_name": "Feedback LLM"})
    
    try:
        feedback = response.strip()
    except:
        feedback = "Could not extract feedback."

    return feedback