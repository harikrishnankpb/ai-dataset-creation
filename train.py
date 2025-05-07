import os
import json
from docx import Document
import pdfplumber
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# I need to adjust this accodingly
chunk_size = 800
chunk_overlap=100

# This is the model for question generation and answer from a text chunk
qg_model="valhalla/t5-small-qa-qg-hl"
qa_model="deepset/minilm-uncased-squad2"

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
def split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Step 3: Generate QA pairs
def generate_qa_pairs(text_chunks, qg_model=qg_model, qa_model=qa_model):
    qa_pairs = []
    
    # Question generation pipeline
    qg_pipeline = pipeline("text2text-generation", 
                          model=qg_model, 
                          max_length=512,
                          num_return_sequences=3,  # Generate multiple questions per chunk
                          do_sample=True,
                          temperature=0.7)  # Add some randomness for variety

    # QA answering pipeline
    qa_pipeline = pipeline("question-answering", model=qa_model)

    for chunk in text_chunks:
        # Generate multiple sets of questions with different prompts
        prompts = [
            f"generate questions: {chunk}",
            f"ask questions about this text: {chunk}",
            f"create detailed questions from this context: {chunk}"
        ]
        
        all_questions = set()  # Use set to avoid duplicates
        for prompt in prompts:
            responses = qg_pipeline(prompt)
            for response in responses:
                questions = response['generated_text'].split("\n")
                all_questions.update([q.strip() for q in questions if "?" in q and len(q.strip()) > 10])

        for q in all_questions:
            # Generate answer for each question
            try:
                result = qa_pipeline(question=q, context=chunk)
                answer = result.get('answer', '').strip()
                confidence = result.get('score', 0)
                
                # Only keep answers with decent confidence and length
                if answer and confidence > 0.2 and len(answer) > 5:
                    qa_pairs.append({
                        "prompt": q,
                        "completion": answer,
                        "confidence": confidence
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

# Main function
def main():
    input_file = "company.pdf"  # File can be pdf or doc
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
