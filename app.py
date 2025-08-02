import streamlit as st
import os
import re
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
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
# Use cloud Qdrant.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

if QDRANT_API_KEY:
    # Use cloud Qdrant
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    st.success("✅ Connected to Qdrant Cloud")
else:
    # Use local Qdrant (for development)
    qdrant_client = QdrantClient(url=QDRANT_URL)
    st.info("ℹ️ Using local Qdrant (for cloud deployment, set QDRANT_URL and QDRANT_API_KEY)")

# Set up the LLM
try:
    client = InferenceClient(
        provider="novita",
        api_key=os.getenv("HF_TOKEN"),
    )
    
    # Create a custom LLM wrapper for llama-index
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
    st.success("✅ Using GLM-4.5 via Novita Inference Client")
except Exception as e:
    st.error(f"❌ GLM-4.5 setup failed: {e}")
    Settings.llm = None

# ----------------- EMBEDDINGS -----------------
try:
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    Settings.embed_model = embed_model
    st.success("✅ HuggingFace embeddings loaded successfully")
except Exception as e:
    st.error(f"❌ Embedding model setup failed: {e}")
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

    # Set up global proxy for requests
    proxies = {
        "http": f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001",
        "https": f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001",
    }
    
    # Patch requests to use proxies
    requests.Session.proxies = proxies

    try:
        # Use the correct method name
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"❌ Error fetching transcript: {e}")
        return None

def loadYoutubeURL(url):
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL")
        return

    with st.spinner("⏳ Loading Index..."):
        try:
            # Clear all existing collections first
            collections = [c.name for c in qdrant_client.get_collections().collections]
            for collection_name in collections:
                try:
                    qdrant_client.delete_collection(collection_name)
                    st.info(f"🗑️ Cleared existing collection: {collection_name}")
                except Exception as e:
                    st.warning(f"⚠️ Could not clear collection {collection_name}: {e}")
            
            # Create new collection for this video
            collection_name = f"yt_{video_id}"
            vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

            st.info("🔄 Creating new index in Qdrant...")
            transcript = get_youtube_transcript(video_id)
            if transcript:
                document = Document(text=transcript)
                index = VectorStoreIndex.from_documents([document], vector_store=vector_store)
                st.success("💾 Index stored in Qdrant!")
            else:
                st.error("❌ No transcript available")
                return

            st.session_state["chat_engine"] = index.as_chat_engine(
                chat_mode="condense_question", streaming=True, verbose=True
            )
            st.success("✅ Video processed successfully!")

        except Exception as e:
            st.error(f"❌ Error processing video: {e}")


# ----------------- UI -----------------
st.title("🎥 YouTube Video RAG Chat (GLM-4.5 + Qdrant)")

with st.sidebar:
    urlTextValue = st.text_input(label="🔗 YouTube URL")
    if st.button(label="Load URL"):
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
                    st.write(response.response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.response}
                    )
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
            else:
                st.warning("Please load a YouTube video first!")
