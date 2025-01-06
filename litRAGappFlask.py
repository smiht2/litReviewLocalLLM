# app.py

import os
import PyPDF2
import faiss
from flask import Flask, request, render_template, jsonify
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key'

# Utility: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Utility: Split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Utility: Build FAISS index
def build_faiss_index(chunks: List[str]):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(chunks).toarray()
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, vectorizer

# Utility: Retrieve relevant chunks
def retrieve_relevant_chunks(question, index, vectorizer, chunks, top_k=3):
    query_vector = vectorizer.transform([question]).toarray()
    _, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

# Utility: Use OpenAI for answering
def answer_question(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant specialized in analyzing research papers."},
            {"role": "user", "content": f"Context: {context} \n\nQuestion: {question}"}
        ]
    )
    return response['choices'][0]['message']['content']

# Route: Upload PDF
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(pdf_path)
        
        # Process PDF
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        index, vectorizer = build_faiss_index(chunks)

        # Answer specific questions
        contribution_question = "What is the contribution of this paper?"
        dataset_question = "What dataset was used for the experiment?"
        
        # Retrieve relevant chunks
        contribution_chunks = retrieve_relevant_chunks(contribution_question, index, vectorizer, chunks)
        dataset_chunks = retrieve_relevant_chunks(dataset_question, index, vectorizer, chunks)
        
        # Generate answers
        contribution_context = " ".join(contribution_chunks)
        dataset_context = " ".join(dataset_chunks)

        contribution_answer = answer_question(contribution_question, contribution_context)
        dataset_answer = answer_question(dataset_question, dataset_context)

        # Return results
        return jsonify({
            'contribution': contribution_answer,
            'dataset': dataset_answer
        })

# Main
if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')
    app.run(debug=True)
