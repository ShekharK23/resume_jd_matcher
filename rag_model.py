from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import re
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from langchain.callbacks.tracers import LangChainTracer
tracer = LangChainTracer()

from langchain_core.runnables import RunnableLambda

llm = OllamaLLM(model="gemma3:1b")

def load_resume(file_path):
    if file_path.endswith(".pdf"):
        return PyMuPDFLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    else:
        raise ValueError("Unsupported file format")

def chunk_by_section(text):

    section_patterns = {
        "summary": r"(summary|objective|about me)",
        "experience": r"(experience|work history|professional experience)",
        "skills": r"(skills|technologies|technical skills)",
        "projects": r"(projects|portfolio)",
        "education": r"(education|academics)",
        "certifications": r"(certifications|qualifications|achievements|endorsements)",
        "strengths": r"(strengths|capabilities|abilities|merits)"
    }

    header_regex = re.compile(
        r'(?P<header>(' + '|'.join(section_patterns.values()) + r'))\s*[:\n]', re.IGNORECASE
    )

    matches = list(header_regex.finditer(text))
    chunks = {}

    for i, match in enumerate(matches):
        header = match.group('header').strip().lower()
        start = match.end()

        section_key = None
        for key, pattern in section_patterns.items():
            if re.fullmatch(pattern, header, re.IGNORECASE):
                section_key = key
                break

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if section_key:
            section_text = text[start:end].strip()
            chunks[section_key] = section_text

    return chunks

def store_embeddings(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_stores = {}

    for section, content in chunks.items():
        docs = [Document(page_content=content, metadata={"section": section})]
        faiss_db = FAISS.from_documents(docs, embedding_model)
        vector_stores[section] = faiss_db

    return vector_stores

# def calculate_similarity(vector_stores, jd_text):
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     query_vector = embedding_model.embed_query(jd_text, callbacks=[tracer], config={"run_name": "embed-query"})

#     matches = {}

#     for section, faiss_db in vector_stores.items():
#         result = faiss_db.similarity_search_by_vector(query_vector, k=1)
#         matches[section] = result[0].page_content if result else ""

#     return matches

def calculate_similarity(vector_stores, jd_text, tracer=None):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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
    import re

    sections_text = "\n\n".join(
        f"[{section}]\n{content}" for section, content in sections_dict.items()
    )

    section_names_list = "\n".join(f"- {section}" for section in sections_dict.keys())

    prompt = f"""
You are a strict and concise evaluator.

You will receive:
1. A resume broken into **PRE-DEFINED sections**.
2. A job description (JD).

Your task:
- Evaluate each resume section **exactly as labeled**.
- Use **ONLY these section names** (do NOT rename or invent new ones):
{section_names_list}

- Assign a score from 0 (poor match) to 10 (perfect match) for each section's relevance to the JD.
- Output in this format ONLY:

SCORES:
<Section Name>: <score>
<Section Name>: <score>
...

DO NOT:
- Generate new sections
- Modify section names
- Provide explanation or commentary

Now evaluate:

RESUME SECTIONS:
{sections_text}

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"
"""

    # response = llm.invoke(prompt, config={"callbacks": [tracer]})
    response = llm.invoke(prompt, config={"callbacks": [tracer], "run_name": "Score LLM"})
    # config={"callbacks": [tracer], "run_name": "llm-invoke"}

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

    prompt = f"""
You are a helpful resume reviewer.

You will be given:
1. A FULL RESUME
2. A JOB DESCRIPTION (JD)

Your task is to analyze the resume **in relation to the JD** and suggest **specific improvements** that would make the resume a stronger match for the JD.

Instructions:
- Only suggest what is MISSING or WEAK in the resume, compared to the JD.
- Be SPECIFIC: mention skills, tools, experience, or domain knowledge that should be added or emphasized.
- DO NOT praise or repeat what's already matching.
- Limit feedback to MAXIMUM 10 bullet points.
- Each point should start with: "- "

RESUME:
\"\"\"{full_resume_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

FEEDBACK:
"""

    response = llm.invoke(prompt, config={"callbacks": [tracer], "run_name": "Feedback LLM"})
    # response = "LLM Call commented"
    try:
        feedback = response.strip()
    except:
        feedback = "Could not extract feedback."

    return feedback

def plot_scores_bar_chart(scores):
 
    fig, ax = plt.subplots(figsize=(10, 6))
    keys = list(scores.keys())
    values = list(scores.values())

    plt.bar(keys, values)
    plt.ylim(0, 10)
    plt.xlabel("Resume Sections")
    plt.ylabel("Matching Score")
    plt.title("Resume Sections vs Matching Scores")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def pie_plot_score_chart(score):

    total_score = 10

    # Data and labels
    data = [score, total_score - score]
    labels = ['Match', 'Remaining']
    colors = ['#4CAF50', '#e0e0e0']

    # Plot pie chart
    fig, ax = plt.subplots(figsize=(4, 1.6), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(
        data,
        labels=labels,
        autopct='%.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4),
        textprops=dict(color="black")
    )

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64
