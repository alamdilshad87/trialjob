from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
    return text.strip()


# Extract text from DOCX
def extract_text_from_docx(file_path):
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        return ""


# Extract text from TXT
def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error extracting text from TXT {file_path}: {e}")
        return ""


# Extract text from CSV
def extract_text_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return ' '.join(df.astype(str).values.flatten())
    except Exception as e:
        logger.error(f"Error extracting text from CSV {file_path}: {e}")
        return ""


# Determine file type and extract text
def extract_text(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    elif ext == 'csv':
        return extract_text_from_csv(file_path)
    return ""


# Home route
@app.route("/")
def home():
    return render_template("resume.html")


# Resume matching route
@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        resumes = []
        resume_names = []

        # Save and extract resumes
        for resume_file in resume_files:
            if resume_file.filename:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                resume_file.save(filename)
                text = extract_text(filename)
                if text.strip():
                    resumes.append(text)
                    resume_names.append(resume_file.filename)

                # Delete the file after processing
                if os.path.exists(filename):
                    os.remove(filename)

        if not resumes or not job_description.strip():
            return render_template('resume.html', message="Please upload valid resumes and enter a job description.")

        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000).fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Get top 5 matching resumes
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [resume_names[i] for i in top_indices]
        similarity_scores = [f"{round(similarities[i] * 100, 2)}%" for i in top_indices]

        return render_template('resume.html', message="Top Matching Resumes:", top_resumes=top_resumes,
                               similarity_scores=similarity_scores)

    return render_template('resume.html')


# Run Flask app
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

