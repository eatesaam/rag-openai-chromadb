# OpenAI Document Search with LLAMA Index

This project is aimed at building a document search system using LLAMA Index, integrating OpenAI's language models for text processing and document retrieval.

## Overview

The application consists of functionalities to add documents to an index and retrieve relevant documents based on user queries. It utilizes LLAMA Index for efficient document storage and retrieval, and OpenAI's models for text processing and reranking.

## Features

- **Document Addition:** Add PDF documents to the search index.
  
- **Document Retrieval:** Retrieve relevant documents based on user queries.
  
- **Semantic Splitting:** Uses semantic splitting for document parsing and embedding.
  
- **Reranking:** Utilizes Colbert Rerank for reranking search results.

## Components

- **Core Components:**
  - `VectorStoreIndex`: Handles indexing of documents using LLAMA Index.
  - `StorageContext`: Manages storage context for LLAMA Index.
  - `Settings`: Configuration settings for the application.
  - `IngestionPipeline`: Pipeline for document ingestion and processing.
- **LLAMA Index Components:**
  - `ChromaVectorStore`: Vector store implementation for LLAMA Index using ChromaDB.
- **OpenAI Integration:**
  - `OpenAI`: Interface for interacting with OpenAI's language models.
- **Text Processing:**
  - `HuggingFaceEmbedding`: Hugging Face embedding model for document embeddings.
- **Document Reading:**
  - `PDFReader`: PDF document reader for extracting text from PDF files.
- **Query Processing:**
  - `SemanticSplitterNodeParser`: Semantic node parser for document splitting.
  - `ColbertRerank`: Reranking module using Colbert model.
