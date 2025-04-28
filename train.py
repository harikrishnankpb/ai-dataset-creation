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
def generate_qa_pairs(text_chunks, model_name="valhalla/t5-small-qa-qg-hl"):
    qa_pipeline = pipeline(
        "text2text-generation",
        model=model_name,
        max_length=512
    )
    print("QA pipeline",qa_pipeline)
    qa_pairs = []
    for chunk in text_chunks:
        prompt = f"generate questions: {chunk}"
        response = qa_pipeline(prompt)[0]['generated_text']
        # print(response)
        # Response may return multiple QAs separated by newlines
        questions_answers = response.split("\n")
        for qa in questions_answers:
            if "?" in qa:
                parts = qa.split("?")
                if len(parts) >= 2:
                    question = parts[0].strip() + "?"
                    answer = parts[1].strip()
                    print("Question:",question)
                    print("Answer:",answer)
                    if question and answer:
                        qa_pairs.append({"prompt": question, "completion": answer})
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
