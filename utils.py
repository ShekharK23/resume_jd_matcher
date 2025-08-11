from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader

import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load Single resume using langchain Data Loaders
def load_resume(file_path):
    if file_path.endswith(".pdf"):
        return PyMuPDFLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    else:
        raise ValueError("Unsupported file format")
    
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
