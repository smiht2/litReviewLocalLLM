import os
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import pandas as pd

# Function: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return []

# Function: Build FAISS index
def build_faiss_index(documents):
    try:
        embeddings = OpenAIEmbeddings()  # Optional: Use OpenAI embeddings for FAISS
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        return None

# Function: Process a single PDF
def process_pdf(pdf_path, retriever, llm, contribution_template, dataset_template):
    try:
        print(f"Processing {os.path.basename(pdf_path)}...")
        pdf_title = os.path.basename(pdf_path)

        # Prepare questions
        contribution_qa = RetrievalQA.from_chain_type(
            retriever=retriever,
            llm=llm,
            chain_type="stuff",
            chain_type_kwargs={"prompt": contribution_template}
        )
        dataset_qa = RetrievalQA.from_chain_type(
            retriever=retriever,
            llm=llm,
            chain_type="stuff",
            chain_type_kwargs={"prompt": dataset_template}
        )

        # Ask questions
        contribution = contribution_qa.run("What is the contribution of this paper?")
        dataset = dataset_qa.run("What dataset was used for the experiment?")

        return {
            "Title of the Paper": pdf_title,
            "Contribution of the Paper": contribution,
            "Dataset Used": dataset
        }
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return {
            "Title of the Paper": os.path.basename(pdf_path),
            "Contribution of the Paper": "Error",
            "Dataset Used": "Error"
        }

# Main Function: Process multiple PDFs
def process_pdfs(input_folder, output_file):
    results = []

    try:
        # Initialize LLM (Ollama's Llama 3.3 model)
        llm = ChatOllama(model="llama3.3")
        contribution_template = PromptTemplate(input_variables=["context"], template="Context: {context}\n\nWhat is the contribution of this paper?")
        dataset_template = PromptTemplate(input_variables=["context"], template="Context: {context}\n\nWhat dataset was used for the experiment?")

        # Iterate over all PDFs
        for filename in os.listdir(input_folder):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(input_folder, filename)

                # Extract text and build FAISS index
                documents = extract_text_from_pdf(pdf_path)
                if not documents:
                    results.append({
                        "Title of the Paper": filename,
                        "Contribution of the Paper": "Error extracting text",
                        "Dataset Used": "Error extracting text"
                    })
                    continue

                vector_store = build_faiss_index(documents)
                if vector_store is None:
                    results.append({
                        "Title of the Paper": filename,
                        "Contribution of the Paper": "Error building index",
                        "Dataset Used": "Error building index"
                    })
                    continue

                retriever = vector_store.as_retriever()

                # Process each PDF and get results
                result = process_pdf(pdf_path, retriever, llm, contribution_template, dataset_template)
                results.append(result)

        # Save results to Excel or CSV
        df = pd.DataFrame(results)
        if output_file.endswith(".csv"):
            df.to_csv(output_file, index=False)
        else:
            df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Unexpected error during processing: {e}")

# Entry Point
if __name__ == "__main__":
    input_folder = "./pdfs"  # Folder containing PDFs
    output_file = "research_paper_analysis.xlsx"  # Output Excel file

    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
    else:
        process_pdfs(input_folder, output_file)
