import os
import json
import argparse
import logging
from typing import List, Dict
from docx import Document
import pdfplumber
from transformers import pipeline
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import concurrent.futures
import re
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qa_generation.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate QA pairs from documents')
    parser.add_argument('--input_file', type=str, required=True, help='Input PDF or DOCX file')
    parser.add_argument('--output_file', type=str, default='training_data.jsonl', help='Output JSONL file')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Text chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=150, help='Overlap between chunks')
    parser.add_argument('--qg_model', type=str, default='valhalla/t5-small-qa-qg-hl', help='Question generation model')
    parser.add_argument('--qa_model', type=str, default='deepset/minilm-uncased-squad2', help='Question answering model')
    parser.add_argument('--min_confidence', type=float, default=0.2, help='Minimum confidence score for QA pairs')
    parser.add_argument('--min_answer_length', type=int, default=5, help='Minimum answer length')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

# Global config initialized from command line args
config = parse_args()

# I need to adjust this accodingly
chunk_size = config.chunk_size
chunk_overlap = config.chunk_overlap

# This is the model for question generation and answer from a text chunk
qg_model = config.qg_model
qa_model = config.qa_model

def get_device():
    if config.use_gpu:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    return "cpu"

# Step 1: Extract text from DOCX or PDF
def extract_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".docx":
            return extract_text_docx(file_path)
        elif ext == ".pdf":
            return extract_text_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise

def extract_text_docx(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if not text:
            logger.warning(f"No text content found in DOCX file: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error processing DOCX file {file_path}: {str(e)}")
        raise

def extract_text_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logger.warning(f"No text extracted from page {page_num}")
        if not text:
            logger.warning(f"No text content found in PDF file: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {str(e)}")
        raise

# Step 2: Split text into manageable chunks
def split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
    """Split text into chunks with improved handling of sentences and sections.
    
    Args:
        text (str): Input text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        List[str]: List of text chunks
    """
    # First, clean up the text
    text = clean_text(text)
    
    # Configure the splitter with better defaults
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ]
    )
    
    # Split the text
    chunks = splitter.split_text(text)
    
    # Post-process chunks
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Clean up the chunk
        chunk = chunk.strip()
        
        # Skip empty chunks
        if not chunk:
            continue
            
        # Ensure chunks end with complete sentences where possible
        if not chunk.endswith('.') and i < len(chunks) - 1:
            # Find the last complete sentence
            last_period = chunk.rfind('.')
            if last_period != -1:
                # Move the remainder to the next chunk
                remainder = chunk[last_period + 1:].strip()
                chunk = chunk[:last_period + 1].strip()
                if i + 1 < len(chunks):
                    chunks[i + 1] = remainder + " " + chunks[i + 1]
        
        processed_chunks.append(chunk)
    
    
    return processed_chunks

def process_chunk(args):
    chunk_idx, chunk = args
    chunk_qa_pairs = []
    
    # Initialize models inside the worker process
    device = "cpu"  # Force CPU for worker processes
    qg_pipeline = pipeline("text2text-generation", model=qg_model, num_return_sequences=1, device=device)
    qa_pipeline = pipeline("question-answering", model=qa_model, device=device)
    
    prompts = [
        f"Generate specific factual questions from this text: {chunk}",
        f"What are the key points someone should understand from this text? Form as questions: {chunk}",
        f"Create analytical questions that test deep understanding of this content: {chunk}",
        f"Generate questions about relationships and connections in this text: {chunk}",
        f"Form questions about the main concepts and definitions from this text: {chunk}",
        f"What questions would verify someone's comprehension of these details: {chunk}",
        f"You are a corporate analyst. Write 5 factual questions based on this content:{chunk}",
        f"As a product trainer, what are some questions you'd ask to test understanding of this material?{chunk}",
        f"Summarize this text by forming 5 'what', 'why', and 'how' questions:{chunk}",
        f"Create 5 interview-style questions from this:{chunk}",
        f"Generate 5 questions someone might ask when learning this for the first time:{chunk}",
        f"What questions can help uncover the purpose, features, and limitations of the subject in this text?{chunk}",
    ]
    
    all_questions = set()
    for prompt in prompts:
        try:
            responses = qg_pipeline(prompt)
            for response in responses:
                questions = response['generated_text'].split("\n")
                all_questions.update([q.strip() for q in questions if "?" in q and len(q.strip()) > 10])
        except Exception as e:
            logger.error(f"Error generating questions for prompt in chunk {chunk_idx}: {str(e)}")
            continue

    for q in all_questions:
        try:
            result = qa_pipeline(question=q, context=chunk)
            answer = result.get('answer', '').strip()
            confidence = result.get('score', 0)
            
            if answer and confidence > config.min_confidence and len(answer) > config.min_answer_length:
                chunk_qa_pairs.append({
                    "prompt": q,
                    "completion": answer,
                    "confidence": confidence,
                    "chunk_index": chunk_idx,
                    "context": chunk
                })
        except Exception as e:
            logger.error(f"Error answering question '{q}' in chunk {chunk_idx}: {str(e)}")
            continue
            
    return chunk_qa_pairs

def clean_text(text):
    """Clean up text for better processing.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR issues
    text = text.replace('|', 'I')  # Common OCR mistake
    text = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', '\n', text)  # Add newlines after sentences
    
    # Remove any non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    return text.strip()

def generate_qa_pairs(text_chunks):
    qa_pairs = []
    total_chunks = len(text_chunks)
    
    try:
        logger.info(f"Initializing with {config.num_workers} workers")
        
        # Prepare arguments for parallel processing
        chunk_args = [(i, chunk) for i, chunk in enumerate(text_chunks, 1)]

        if config.num_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_workers) as executor:
                futures = [executor.submit(process_chunk, arg) for arg in chunk_args]
                
                with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            chunk_results = future.result()
                            qa_pairs.extend(chunk_results)
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                            pbar.update(1)
                            continue
        else:
            # Single process mode
            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                for arg in chunk_args:
                    try:
                        chunk_results = process_chunk(arg)
                        qa_pairs.extend(chunk_results)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        pbar.update(1)
                        continue

        logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")
        return qa_pairs
        
    except Exception as e:
        logger.error(f"Fatal error in QA pair generation: {str(e)}")
        raise

def save_jsonl(qa_pairs, output_file):
    try:
        with tqdm(total=len(qa_pairs), desc="Saving QA pairs") as pbar:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in qa_pairs:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
                    pbar.update(1)
        logger.info(f"Successfully saved {len(qa_pairs)} QA pairs to {output_file}")
    except Exception as e:
        logger.error(f"Error saving QA pairs to {output_file}: {str(e)}")
        raise

# Main function
def main():
    try:
        logger.info("Starting QA pair generation process")
        logger.info(f"Input file: {config.input_file}")
        logger.info(f"Output file: {config.output_file}")

        logger.info("Extracting text...")
        text = extract_text(config.input_file)

        logger.info("Splitting text...")
        chunks = split_text(text)
        logger.info(f"Generated {len(chunks)} chunks")

        logger.info("Generating QA pairs...")
        qa_pairs = generate_qa_pairs(chunks)
        logger.info(f"Generated {len(qa_pairs)} QA pairs")

        logger.info("Saving to JSONL...")
        save_jsonl(qa_pairs, config.output_file)

        logger.info(f"Process completed successfully. Dataset saved to {config.output_file}")
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
