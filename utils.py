# Imports 
from rank_bm25 import BM25Okapi  # For BM25 ranking
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split large documents
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF vector creation
from nltk.tokenize.toktok import ToktokTokenizer  # Lightweight tokenizer
from sklearn.metrics.pairwise import cosine_similarity  # To compute TF-IDF cosine similarity
import numpy as np  # For numerical operations
from langchain_groq import ChatGroq  # Groq-backed LLM interface
from langchain.prompts import PromptTemplate  # LangChain prompt template
from langchain.chains import LLMChain  # LangChain chain
from dotenv import load_dotenv  # Load environment variables
import os 
import time
import pathlib
import PyPDF2
import streamlit as st

# Tokenizer instance for text processing 
tokenizer = ToktokTokenizer()

# Normalizes a list to 0–1 range 
def normalize_list(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0 for _ in values]  # Prevent divide-by-zero
    return [(x - min_val) / (max_val - min_val) for x in values]

# Splits list of documents into small overlapping chunks 
def text_split(documents, chunk_size=200, chunk_overlap=20):
    all_documents_concatenated = "\n".join(documents)  # Combine all docs into one large string
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_text(all_documents_concatenated)
    return text_chunks  # Returns a list of small text segments

# BM25 Search: uses tokenized query and corpus to return BM25 relevance scores 
def BM25_search(query, tokenized_corpus):
    bm25 = BM25Okapi(tokenized_corpus)  # Initialize BM25 with tokenized corpus
    tokenized_query = tokenizer.tokenize(query.lower())  # Tokenize the query
    doc_scores = bm25.get_scores(tokenized_query)  # Get BM25 scores for query
    return normalize_list(doc_scores)  # Normalize scores to 0–1

# TF-IDF Search: computes cosine similarity of query with corpus chunks 
def TF_IDF(query, chunks):
    tfidf_vectorizer = TfidfVectorizer()  # Initialize TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)  # Fit corpus
    query_vector = tfidf_vectorizer.transform([query])  # Transform query
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()  # Flatten result to 1D array
    return cosine_similarities  # Higher = more relevant

# Hybrid Search: combines TF-IDF and BM25 with weighted average 
def hybrid_search(query, top_n, chunks, tokenized_corpus, alpha):
    score1 = TF_IDF(query, chunks)  # TF-IDF score vector
    score2 = BM25_search(query, tokenized_corpus)  # BM25 score vector

    # Weighted combination of TF-IDF and BM25 scores
    score_hybrid = alpha * np.array(score1) + (1 - alpha) * np.array(score2)

    # Get top-N documents with highest hybrid scores
    top_n_docs = sorted(enumerate(score_hybrid), key=lambda x: x[1], reverse=True)[:top_n]

    # Concatenate top-N relevant documents into one string
    relevant_docs = ".".join([chunks[idx] for idx, score in top_n_docs])
    
    return relevant_docs  # Return final relevant context to feed into LLM
