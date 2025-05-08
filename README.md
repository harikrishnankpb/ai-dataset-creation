# Document QA Dataset Generator

This tool automatically generates question-answer pairs from PDF and DOCX documents using state-of-the-art language models. It's designed to create high-quality training datasets for question-answering systems.

## Features

- Supports both PDF and DOCX document formats
- Intelligent text chunking with configurable size and overlap
- Parallel processing for faster generation
- Progress tracking with detailed logging
- Configurable quality thresholds for QA pairs
- Comprehensive error handling and reporting

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python train.py --input_file your_document.pdf --output_file qa_pairs.jsonl
```

Advanced options:
```bash
python train.py \
    --input_file your_document.pdf \
    --output_file qa_pairs.jsonl \
    --chunk_size 1000 \
    --chunk_overlap 150 \
    --qg_model valhalla/t5-small-qa-qg-hl \
    --qa_model deepset/minilm-uncased-squad2 \
    --min_confidence 0.2 \
    --min_answer_length 5 \
    --num_workers 4
```

## Parameters

- `--input_file`: Path to the input document (PDF or DOCX)
- `--output_file`: Path for the output JSONL file (default: training_data.jsonl)
- `--chunk_size`: Size of text chunks for processing (default: 1000)
- `--chunk_overlap`: Overlap between consecutive chunks (default: 150)
- `--qg_model`: Question generation model to use (default: valhalla/t5-small-qa-qg-hl)
- `--qa_model`: Question answering model to use (default: deepset/minilm-uncased-squad2)
- `--min_confidence`: Minimum confidence score for QA pairs (default: 0.2)
- `--min_answer_length`: Minimum answer length in characters (default: 5)
- `--num_workers`: Number of parallel workers (default: number of CPU cores)

## Output Format

The tool generates a JSONL file where each line contains a JSON object with:
- `prompt`: The generated question
- `completion`: The extracted answer
- `confidence`: Model's confidence score for the answer
- `chunk_index`: Source text chunk index

## Logging

The tool creates a detailed log file (`qa_generation.log`) containing:
- Progress information
- Warning messages for empty text chunks
- Error messages for failed processing attempts
- Final statistics

## Error Handling

The tool includes comprehensive error handling for:
- File not found errors
- Unsupported file formats
- Text extraction failures
- Model processing errors
- File writing errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 


## Environment setup

# Setup a virtual environment
python3 -m venv data_set
source data_set/bin/activate

# Install the required packages
pip install -r requirements.txt

# Destroy the virtual environment
deactivate
rm -rf data-set

# Remove model from local 
rm -rf ~/.cache/huggingface/