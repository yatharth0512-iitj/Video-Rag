import streamlit as st
import os
import re
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import CustomLLM, LLMMetadata
from llama_index.core.llms import CompletionResponse
from huggingface_hub import InferenceClient

# ----------------- LOAD ENV -----------------
load_dotenv()
HF_API_URL = "https://api-inference.huggingface.co/models/zai-org/GLM-4.5"
HF_TOKEN = os.getenv("HF_TOKEN")

# ----------------- CUSTOM LLM -----------------
class HFRemoteLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=4096, num_output=512)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
        response = requests.post(HF_API_URL, headers=headers, json=payload)

        try:
            result = response.json()
            if isinstance(result, list):
                text_output = result[0]["generated_text"]
            elif "generated_text" in result:
                text_output = result["generated_text"]
            else:
                text_output = str(result)
            return CompletionResponse(text=text_output)
        except Exception as e:
            return CompletionResponse(text=f"Error from Hugging Face API: {e}")

# ----------------- QDRANT -----------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

if QDRANT_API_KEY:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant_client = QdrantClient(url=QDRANT_URL)

# ----------------- LLM SETUP -----------------
try:
    client = InferenceClient(
        provider="novita",
        api_key=HF_TOKEN,
    )

    class GLM4LLM(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(context_window=4096, num_output=512)

        def complete(self, prompt: str, **kwargs) -> CompletionResponse:
            try:
                completion = client.chat.completions.create(
                    model="zai-org/GLM-4.5",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.7
                )
                text_output = completion.choices[0].message.content
                return CompletionResponse(text=text_output)
            except Exception as e:
                return CompletionResponse(text=f"Error: {e}")

        def stream_complete(self, prompt: str, **kwargs):
            response = self.complete(prompt, **kwargs)
            yield response

    llm = GLM4LLM()
    Settings.llm = llm
except Exception as e:
    Settings.llm = None

# ----------------- EMBEDDINGS -----------------
try:
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    Settings.embed_model = embed_model
except Exception as e:
    Settings.embed_model = None

# ----------------- HELPERS -----------------
def extract_video_id(url):
    pattern = r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_youtube_transcript(video_id):
    SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY")
    if not SCRAPERAPI_KEY:
        st.error("❌ Missing ScraperAPI key!")
        return None

    proxy_url = f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"

    # Monkey-patch requests.get to use the proxy
    original_get = requests.get
    def proxy_get(url, **kwargs):
        kwargs["proxies"] = {"http": proxy_url, "https": proxy_url}
        return original_get(url, **kwargs)

    requests.get = proxy_get

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        return " ".join([entry.text for entry in transcript])
    except TranscriptsDisabled:
        st.error("❌ Transcripts are disabled for this video.")
        return None
    except NoTranscriptFound:
        st.error("❌ No transcript available for this video.")
        return None
    except VideoUnavailable:
        st.error("❌ This video is unavailable or private.")
        return None
    except Exception as e:
        st.error(f"❌ Error fetching transcript: {e}")
        return None
    finally:
        requests.get = original_get  # Restore original function


def loadYoutubeURL(url):
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL")
        return

    with st.spinner("⏳ Loading Index..."):
        try:
            collections = [c.name for c in qdrant_client.get_collections().collections]
            for collection_name in collections:
                try:
                    qdrant_client.delete_collection(collection_name)
                    st.info(f"🗑️ Cleared existing collection: {collection_name}")
                except Exception as e:
                    st.warning(f"⚠️ Could not clear collection {collection_name}: {e}")

            collection_name = f"yt_{video_id}"
            vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

            transcript = get_youtube_transcript(video_id)
            if transcript:
                document = Document(text=transcript)
                index = VectorStoreIndex.from_documents([document], vector_store=vector_store)
            else:
                st.error("❌ No transcript available")
                return

            st.session_state["chat_engine"] = index.as_chat_engine(
                chat_mode="condense_question", streaming=True, verbose=True
            )
        except Exception as e:
            st.error(f"❌ Error processing video: {e}")

# ----------------- UI -----------------

# Main title
st.title("🎥 YouTube Video RAG")
st.markdown("*Transform any YouTube video into an interactive AI assistant*")

# What is RAG section
st.header("🚀 What is RAG (Retrieval-Augmented Generation)?")
st.markdown("RAG is an AI technique that combines the power of large language models with external knowledge sources.")

# RAG components in columns
col1, col2 = st.columns(2)
with col1:
    st.info("🔍 **Retrieval**\n\nFinding relevant information from video transcripts")
with col2:
    st.info("🧠 **Generation**\n\nAI generates contextual responses")

# How it works section
st.header("📋 How it works with YouTube videos:")
col1, col2, col3 = st.columns(3)
with col1:
    st.success("1️⃣ **Upload**\n\nPaste any YouTube video URL")
with col2:
    st.warning("2️⃣ **Process**\n\nSystem extracts and analyzes transcript")
with col3:
    st.info("3️⃣ **Chat**\n\nAsk questions and get intelligent responses")

# Benefits section
st.header("✨ Benefits:")
col1, col2 = st.columns(2)
with col1:
    st.success("✅ **Accurate Answers**\n\nResponses based on actual video content")
    st.info("✅ **Context Awareness**\n\nAI understands video context and topics")
with col2:
    st.error("✅ **Less Hallucinations**\n\nInformation comes directly from source")
    st.warning("✅ **Interactive Learning**\n\nPerfect for educational content")

# Divider
st.markdown("---")

# URL input section
st.header("🎬 Ready to Get Started?")
st.markdown("Paste your YouTube URL below and transform it into an interactive AI assistant")

# Center the URL input
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    urlTextValue = st.text_input(
        label="🔗 YouTube URL", 
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter any YouTube video URL to get started"
    )
    
    if st.button(
        label="🚀 Load Video & Start Chatting", 
        use_container_width=True,
        type="primary"
    ):
        if urlTextValue:
            loadYoutubeURL(urlTextValue)
        else:
            st.error("Please enter a valid YouTube URL")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            if "chat_engine" in st.session_state:
                try:
                    response = st.session_state["chat_engine"].chat(prompt)
                    # Clean the response to remove <think> tags
                    clean_response = response.response.replace('<think>', '').replace('</think>', '').strip()
                    st.write(clean_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": clean_response}
                    )
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
            else:
                st.warning("Please load a YouTube video first!")
