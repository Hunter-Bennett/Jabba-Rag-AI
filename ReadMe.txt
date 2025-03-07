# Document Retrieval and AI Answering System

This project allows you to store documents (PDFs and text files), create a searchable FAISS index, and generate AI responses to queries based on the content of those documents. It utilizes **FAISS** for efficient document search, **Sentence-Transformers** for embeddings, and **Ollama** to generate responses from the context.

## Features

- **Document Loading**: Supports PDF and text files stored in a folder.
- **Document Chunking**: Splits documents into smaller chunks to improve search and response quality.
- **FAISS Index**: Efficient vector-based search system to find relevant document chunks.
- **AI Answer Generation**: Generates answers based on retrieved document chunks using an AI model (configured with Ollama).

## Prerequisites

- Python 3.x
- Ollama Model (e.g., `gemma2` or another local model)

## Installation

1. Clone this repository to your local machine.

2. Install dependencies:
   - Create a virtual environment:
     ```bash
     python -m venv .venv
     ```
   - Activate the virtual environment:
     - **Windows**: 
       ```bash
       .\.venv\Scripts\activate
       ```
     - **Linux/macOS**:
       ```bash
       source .venv/bin/activate
       ```
   - Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```

3. Download and set up an **Ollama model** (like `gemma2`). Follow the official instructions from Ollama's website to install and run the required model.

## Setup and Usage

1. Place your documents (PDF or text files) in the `./docs` folder.

2. Run the script:
   ```bash
   python Main.py