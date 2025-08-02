# YouTube Video RAG Chat

A Streamlit application that processes YouTube videos and allows users to ask questions about the video content using RAG (Retrieval-Augmented Generation).

## Features

- 🎥 Process YouTube videos and extract transcripts
- 🤖 Use GLM-4.5 via HuggingFace for intelligent responses
- 🗄️ Store embeddings in Qdrant Cloud vector database
- 🧹 Single video storage (clears previous video when new one is loaded)
- 💬 Interactive chat interface

## Environment Variables

- `HF_TOKEN`: Your HuggingFace API token
- `QDRANT_URL`: Your Qdrant Cloud cluster URL
- `QDRANT_API_KEY`: Your Qdrant Cloud API key

## Deployment

This app is ready for deployment on:
- Railway
- Streamlit Cloud
- Render
- Heroku

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
``` 