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
- `SCRAPERAPI_KEY`: Your ScraperAPI key for YouTube transcript access

## Cloud Deployment

This app is ready for deployment on:
- Railway
- Streamlit Cloud
- Render
- Heroku

### Deployment Steps:

1. **Fork/Clone this repository**
2. **Set up environment variables** in your cloud platform:
   - `HF_TOKEN`: Get from [HuggingFace](https://huggingface.co/settings/tokens)
   - `QDRANT_URL`: Get from [Qdrant Cloud](https://cloud.qdrant.io/)
   - `QDRANT_API_KEY`: Get from [Qdrant Cloud](https://cloud.qdrant.io/)
   - `SCRAPERAPI_KEY`: Get from [ScraperAPI](https://www.scraperapi.com/)

3. **Deploy to your chosen platform**

### Troubleshooting Cloud Deployment:

- **Import Errors**: The app uses `youtube_transcript_api._errors` instead of `youtube_transcript_api.exceptions`
- **Python Version**: Uses Python 3.12.0 (specified in runtime.txt)
- **Dependencies**: All dependencies are listed in requirements.txt without version constraints for better cloud compatibility

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Test imports
python test_imports.py

# Run the app
streamlit run app.py
```

## Usage

1. Enter a YouTube URL in the sidebar
2. Click "Load URL" to process the video
3. Ask questions about the video content in the chat interface 