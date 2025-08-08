from flask import Flask, render_template, request
from rag_model import load_resume, chunk_by_section, store_embeddings, calculate_similarity, get_resume_level_feedback, plot_scores_bar_chart, get_scores_for_all_sections, pie_plot_score_chart
import traceback
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        jd_text = request.form['jd_text']
        resume_file = request.files['resume_file']

        filename = secure_filename(resume_file.filename)
        temp_path = os.path.join("temp_resumes", filename)
        os.makedirs("temp_resumes", exist_ok=True)
        resume_file.save(temp_path)

        pages = load_resume(temp_path)
        text = "\n".join([page.page_content for page in pages])

        chunks = chunk_by_section(text)
        # print("Chunks:", chunks)
        vector_store = store_embeddings(chunks)
        best_matches = calculate_similarity(vector_store, jd_text)

        scores = get_scores_for_all_sections(best_matches, jd_text)
        feedback = get_resume_level_feedback(text, jd_text)

        # print("Scores:", scores)
        if not scores:
            raise ValueError("No scores generated from get_llm_feedback.")

        final_score = sum(scores.values()) / len(scores)
        final_score = round(final_score, 2)
        os.remove(temp_path)

        chart = plot_scores_bar_chart(scores)
        pie = pie_plot_score_chart(final_score)

        return render_template('result.html', piechart=pie, score=final_score, scores=scores, explaination=feedback, chart=chart)

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {e}", 500

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def Register():
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)