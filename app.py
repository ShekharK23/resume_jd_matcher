from flask import Flask, render_template, request, jsonify
from rag_model import chunk_by_section, calculate_similarity, get_resume_level_feedback, get_scores_for_all_sections
from rag_model_mass import calculate_resume_similarity, explain_match
from vectorDB import store_embeddings, store_embeddings_mass
from utils import load_resume, plot_scores_bar_chart, pie_plot_score_chart
import traceback
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def Register():
    return render_template('register.html')

@app.route('/mass_index')
def mass_index():
    return render_template('bulk_index.html')

# @app.route('/result', methods=['POST'])
# def result():
#     try:
#         jd_text = request.form['jd_text']
#         resume_file = request.files['resume_file']

#         filename = secure_filename(resume_file.filename)
#         temp_path = os.path.join("temp_resumes", filename)
#         os.makedirs("temp_resumes", exist_ok=True)
#         resume_file.save(temp_path)

#         pages = load_resume(temp_path)
#         text = "\n".join([page.page_content for page in pages])

#         chunks = chunk_by_section(text)
#         # print("Chunks:", chunks)
#         vector_store = store_embeddings(chunks)
#         best_matches = calculate_similarity(vector_store, jd_text)

#         scores = get_scores_for_all_sections(best_matches, jd_text)

#         chart = plot_scores_bar_chart(scores)
#         pie = pie_plot_score_chart(final_score)

#         feedback = get_resume_level_feedback(text, jd_text)

#         # print("Scores:", scores)
#         if not scores:
#             raise ValueError("No scores generated from get_llm_feedback.")

#         final_score = sum(scores.values()) / len(scores)
#         final_score = round(final_score, 2)
#         os.remove(temp_path)


#         return render_template('result.html', piechart=pie, score=final_score, scores=scores, explaination=feedback, chart=chart)

#     except Exception as e:
#         traceback.print_exc()
#         return f"An error occurred: {e}", 500

@app.route('/result', methods=['POST'])
def result():
    try:
        jd_text = request.form['jd_text']
        resume_file = request.files['resume_file']

        filename = secure_filename(resume_file.filename)
        temp_path = os.path.join("temp_resumes", filename)
        os.makedirs("temp_resumes", exist_ok=True)
        resume_file.save(temp_path)

        # Step 1: Load and preprocess
        pages = load_resume(temp_path)
        text = "\n".join([page.page_content for page in pages])
        chunks = chunk_by_section(text)

        # Step 2: Vector store & similarity
        vector_store = store_embeddings(chunks)
        best_matches = calculate_similarity(vector_store, jd_text)

        # Step 3: Scoring
        scores = get_scores_for_all_sections(best_matches, jd_text)
        if not scores:
            raise ValueError("No scores generated from get_llm_feedback.")

        final_score = round(sum(scores.values()) / len(scores), 2)

        # Step 4: Charts
        chart = plot_scores_bar_chart(scores)
        pie = pie_plot_score_chart(final_score)

        os.remove(temp_path)

        # Step 5: NOW run feedback call at the end
        # feedback = get_resume_level_feedback(text, jd_text)

        # Step 6: Render
        return render_template(
            'result.html',
            piechart=pie,
            score=final_score,
            scores=scores,
            # explaination=feedback,
            chart=chart,
            resume_text=text,
            jd_text=jd_text
        )

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {e}", 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        jd_text = request.json.get('jd_text')
        resume_text = request.json.get('resume_text')

        feedback = get_resume_level_feedback(resume_text, jd_text)
        return {"feedback": feedback}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

@app.route('/shortlist', methods=['POST'])
def shortlist_resumes():
    try:
        # Get JD text and number of top resumes to shortlist
        jd_text = request.form['jd_text']
        top_n = int(request.form.get('top_n', 5))

        # Get uploaded resumes
        uploaded_files = request.files.getlist('resume_files')
        if not uploaded_files:
            raise ValueError("No resumes uploaded.")

        os.makedirs("temp_resumes", exist_ok=True)

        resumes = []
        for resume_file in uploaded_files:
            filename = secure_filename(resume_file.filename)
            temp_path = os.path.join("temp_resumes", filename)
            resume_file.save(temp_path)

            # Load and join content
            pages = load_resume(temp_path)
            text = "\n".join([page.page_content for page in pages])

            resumes.append({"file_name": filename, "content": text})

        # Store embeddings (one FAISS DB for all resumes)
        faiss_db = store_embeddings_mass(resumes)

        # Get top matches
        matches = calculate_resume_similarity(faiss_db, jd_text, top_n=top_n)

        # Remove temp files
        for resume_file in uploaded_files:
            file_path = os.path.join("temp_resumes", secure_filename(resume_file.filename))
            if os.path.exists(file_path):
                os.remove(file_path)

        # Render results page (without explanations)
        return render_template(
            'bulk_result.html',
            jd_text=jd_text,
            matches=matches
        )

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {e}", 500

@app.route('/explain_match', methods=['POST'])
def explain_match_route():
    try:
        jd_text = request.form['jd_text']
        resume_content = request.form['resume_content']

        explanation = explain_match(resume_content, jd_text)

        return jsonify({
            "status": "success",
            "explanation": explanation
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)