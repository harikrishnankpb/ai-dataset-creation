import os
import json
from docx import Document
import pdfplumber
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Extract text from DOCX or PDF
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return extract_text_docx(file_path)
    elif ext == ".pdf":
        return extract_text_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_text_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Step 2: Split text into manageable chunks
def split_text(text, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Step 3: Generate QA pairs
def generate_qa_pairs(text_chunks, qg_model="valhalla/t5-small-qa-qg-hl", qa_model="deepset/minilm-uncased-squad2"):
    qa_pairs = []
    
    # Question generation pipeline
    qg_pipeline = pipeline("text2text-generation", model=qg_model, max_length=512)

    # QA answering pipeline
    qa_pipeline = pipeline("question-answering", model=qa_model)

    for chunk in text_chunks:
        # Generate questions
        prompt = f"generate questions: {chunk}"
        response = qg_pipeline(prompt)[0]['generated_text']
        questions = response.split("\n")

        for q in questions:
            q = q.strip()
            if "?" in q:
                # Generate answer for each question
                try:
                    result = qa_pipeline(question=q, context=chunk)
                    answer = result.get('answer', '').strip()
                    if answer:
                        qa_pairs.append({
                            "prompt": q,
                            "completion": answer
                        })
                except Exception as e:
                    print(f"Error answering question '{q}': {e}")
                    continue
    return qa_pairs

# Step 4: Save QA pairs to .jsonl
def save_jsonl(qa_pairs, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in qa_pairs:
            json.dump(item, f)
            f.write('\n')

# Runner
def main():
    input_file = "company.pdf"  # Replace with your input (pdf or docx)
    output_jsonl = 'training_data.jsonl'

    print("Extracting text...")
    text = extract_text(input_file)

    print("Splitting text...")
    chunks = split_text(text)

    print(f"Generated {len(chunks)} chunks.")

    print("Generating QA pairs...")
    qa_pairs = generate_qa_pairs(chunks)

    print(f"Generated {len(qa_pairs)} QA pairs.")

    print("Saving to JSONL...")
    save_jsonl(qa_pairs, output_jsonl)

    print(f"Done! Dataset saved to {output_jsonl}")

if __name__ == "__main__":
    main()
