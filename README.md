# Langchain PDF Chatbot with Streamlit

A conversational chatbot powered by Langchain and powerful models like Google's Gemini and OpenAI, designed to provide a seamless chat interface for querying information from multiple PDF documents.



## 📜 Overview

This project provides an intuitive web interface built with **Streamlit** where you can upload multiple PDFs and ask questions about their content. The chatbot uses natural language processing and retrieval-augmented generation (RAG) to understand your queries, find relevant information within the documents, and provide accurate, context-aware answers.

It leverages powerful language models for conversational abilities and state-of-the-art embedding models to create a searchable knowledge base from your documents.

***

## ✨ Features

* **Interactive UI**: Simple and user-friendly interface powered by Streamlit.
* **Multiple PDF Support**: Upload and chat with one or more PDF documents simultaneously.
* **Conversational Memory**: The chatbot remembers previous questions and answers in a conversation for better context.
* **Powered by Top LLMs**: Integrates with powerful language models like Google Gemini & OpenAI.
* **Efficient Text Processing**: Automatically extracts text from PDFs, splits it into optimized chunks, and indexes it for fast retrieval.
* **Open Source**: Built with leading open-source libraries like Langchain and FAISS.

***

## ⚙️ How It Works

The application follows a retrieval-augmented generation (RAG) pipeline:

1.  **PDF Processing**: When you upload PDFs, the application extracts the text content from each file.
2.  **Text Chunking**: The extracted text is split into smaller, semantically meaningful chunks. This is crucial for efficient and accurate retrieval.
3.  **Embedding Creation**: Each text chunk is converted into a numerical vector (embedding) using an embedding model from Google or OpenAI.
4.  **Vector Storage**: These embeddings are stored in a FAISS vector store, which acts as a searchable index or a knowledge base.
5.  **Conversational Retrieval**: When you ask a question, the chatbot queries the vector store to find the most relevant text chunks. These chunks, along with your question and the conversation history, are passed to a large language model (like Google's Gemini or OpenAI's GPT) to generate a human-like answer.

***

## 🛠️ Tech Stack

* **Frameworks**: Langchain, Streamlit
* **Language Models**: Google Gemini, OpenAI
* **Embeddings**: Google Generative AI Embeddings, OpenAI Embeddings
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **PDF Processing**: PyPDF2

***

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.9 or higher
* An API key from **Google AI Studio** (for Gemini) and/or **OpenAI**

### 1. Clone the Repository

```bash
git clone [https://github.com/bajajlakshit/chatbot.git](https://github.com/bajajlakshit/chatbot.git)
cd chatbot
