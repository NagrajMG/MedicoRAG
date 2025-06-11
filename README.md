# RAG-Powered PDF QA System

A lightweight Retrieval-Augmented Generation (RAG) system that allows you to **upload medical PDFs**, process them, and **ask questions** using a **hybrid TF-IDF + BM25** retrieval and a **Groq-hosted LLM (Qwen-32B)**.



## Features

- Upload any PDF and auto-extract readable text
- Hybrid document retrieval (TF-IDF + BM25)
- Question-answering via LangChain + ChatGroq (Qwen-32B)
- Works for MCQ-based document querying too
- Built with a responsive Streamlit UI
- Start/Stop app control panel



## How It Works

1. **Text Extraction**: PDF is cleaned and non-ASCII characters removed
2. **Chunking**: Text is split into overlapping segments (200 tokens, 20 overlap)
3. **Indexing**: Each chunk is tokenized and stored for retrieval
4. **Hybrid Search**:
   - `score = alpha * TF-IDF + (1 - alpha) * BM25`
5. **LLM Inference**: Top N retrieved chunks fed to Qwen-32B via LangChain for answer generation

---

## Installation
```bash
pip install -r requirements.txt
```

## Directory
```python
MedicoRAG/
│
├── data/
│   └── en/
│       └── extra.pdf   # Uploaded PDF
│       └── extra.txt   # Cleaned text from PDF
├── utils.py            # Text splitting and hybrid search functions
├── engine.py           # LLM setup and RAG logic
├── main.py              # Streamlit frontend
└── README.md
```

## Test site
```bash
npm start
```
or 
```bash
streamlit run main.py
```
---
## License
MIT License

Copyright (c) 2025 Nagraj Gaonkar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
