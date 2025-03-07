import os
import pdfplumber
import spacy
import docx2txt
from flask import Flask, request, jsonify
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

SKILL_SET = {"Python", "JavaScript", "C#", "React", "GCP", "Machine Learning", "Docker", "Kubernetes", "SQL"}

def extract_content(file):
    content = ""
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file.read())) as pdf :
            for page in pdf.pages:
                content += page.extract_text() + "\n"
    
    elif file.filename.endswith(".docx"):
        content = docx2txt.process(io.BytesIO(file.read()))
    
    return content

def extract_skills(text):
    doc = nlp(text)
    skills = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() in SKILL_SET]

    return list(set(skills))

def compare_with_job(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tf = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tf[0:1], tf[1:2])[0][0]*100

    return score

@app.route("/check-resume-score", methods =["POST"])
def check_resume_score():
    if 'resume' not in request.files:
        return jsonify({"error" : "no resume uploaded"}), 400
    
    resume = request.files['resume']
    job_description = request.form.get('job_description', '')

    resume_content = extract_content(resume)
    skills_found = extract_skills(resume_content)
    match_score = compare_with_job(resume_content, job_description)

    return jsonify({
        "match_score" : match_score,
        "skills_found" : skills_found
    })


if __name__ == "__main__":
    app.run(debug=True)

