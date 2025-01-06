import os
import PyPDF2
import faiss
import pandas as pd
from openai import ChatCompletion
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key'

# Utility: Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Utility: Split text into chunks
def split_text_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Utility: Build FAISS index
def build_faiss_index(chunks: List[str]):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(chunks).toarray()
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, vectorizer

# Utility: Retrieve relevant chunks
def retrieve_relevant_chunks(question: str, index, vectorizer, chunks: List[str], top_k: int = 3) -> List[str]:
    query_vector = vectorizer.transform([question]).toarray()
    _, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

# Utility: Use OpenAI for answering
def answer_question(question: str, context: str) -> str:
    response = ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant specialized in analyzing research papers."},
            {"role": "user", "content": f"Context: {context} \n\nQuestion: {question}"}
        ]
    )
    return response['choices'][0]['message']['content']

# Main Function: Process PDFs and generate Excel
def process_pdfs(input_folder: str, output_file: str):
    results = []
    
    # Process all PDFs in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")

            # Extract text and split into chunks
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text_into_chunks(text)

            # Build FAISS index
            index, vectorizer = build_faiss_index(chunks)

            # Questions
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

            # Append results
            results.append({
                "Title of the Paper": filename,
                "Contribution of the Paper": contribution_answer,
                "Dataset Used": dataset_answer
            })

    # Save results to Excel or CSV
    df = pd.DataFrame(results)
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
    else:
        df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# Entry Point
if __name__ == '__main__':
    input_folder = './pdfs'  # Folder containing PDFs
    output_file = 'research_paper_analysis.xlsx'  # Output Excel file
    
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
    else:
        process_pdfs(input_folder, output_file)
