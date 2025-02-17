import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings


# Custom CSS for Perplexity-like styling
st.markdown(
    """
    <style>
    /* General Styling */
    .stApp {
        background-color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput>div>div>input {
        background-color: #f7f7f7;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        padding: 12px 16px;
        font-size: 16px;
        color: #333333;
    }
    .stTextInput>div>div>input::placeholder {
        color: #999999;
    }
    .stButton>button {
        background-color: #10a37f;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0d8a6a;
    }
    /* Response Cards */
    .response-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    .response-card h4 {
        color: #10a37f;
        font-size: 18px;
        margin-bottom: 12px;
    }
    .response-card p {
        color: #333333;
        font-size: 16px;
        margin: 8px 0;
    }
    .response-card a {
        color: #10a37f;
        text-decoration: none;
    }
    .response-card a:hover {
        text-decoration: underline;
    }
    /* Sources Section */
    .sources-section {
        background-color: #f7f7f7;
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
    }
    .sources-section h4 {
        color: #333333;
        font-size: 16px;
        margin-bottom: 12px;
    }
    .sources-section p {
        color: #666666;
        font-size: 14px;
        margin: 4px 0;
    }
    /* Follow-up Questions */
    .follow-up-questions {
        margin: 16px 0;
    }
    .follow-up-questions h4 {
        color: #333333;
        font-size: 16px;
        margin-bottom: 12px;
    }
    .follow-up-questions button {
        background-color: #f7f7f7;
        color: #333333;
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 14px;
        border: 1px solid #e0e0e0;
        margin: 4px;
        cursor: pointer;
    }
    .follow-up-questions button:hover {
        background-color: #e0e0e0;
    }
    /* Dark Mode */
    [data-theme="dark"] .stApp {
        background-color: #1e1e1e;
    }
    [data-theme="dark"] .stTextInput>div>div>input {
        background-color: #2d2d2d;
        border-color: #444444;
        color: #ffffff;
    }
    [data-theme="dark"] .stTextInput>div>div>input::placeholder {
        color: #999999;
    }
    [data-theme="dark"] .response-card {
        background-color: #2d2d2d;
        border-color: #444444;
    }
    [data-theme="dark"] .response-card h4 {
        color: #10a37f;
    }
    [data-theme="dark"] .response-card p {
        color: #ffffff;
    }
    [data-theme="dark"] .sources-section {
        background-color: #2d2d2d;
    }
    [data-theme="dark"] .sources-section h4 {
        color: #ffffff;
    }
    [data-theme="dark"] .sources-section p {
        color: #cccccc;
    }
    [data-theme="dark"] .follow-up-questions button {
        background-color: #2d2d2d;
        color: #ffffff;
        border-color: #444444;
    }
    [data-theme="dark"] .follow-up-questions button:hover {
        background-color: #444444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper function to convert seconds to minutes:seconds format
def seconds_to_min_sec(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02}"


# Initialize the Retriever
class Retriever:
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(persist_directory=db_path, is_persistent=True))
        self.collection = self.client.get_collection(name="video_metadata")
        self.embedding_generator = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve the most relevant chunks based on the user's query.
        """
        query_embedding = self.embedding_generator.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Format the results
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "video_uri": results["metadatas"][0][i]["video_uri"],
                "start_time": results["metadatas"][0][i]["start_time"],
                "text": results["documents"][0][i],
            })

        return retrieved_chunks


# Streamlit App
def main():
    # Title and Header
    st.markdown("<h1 style='text-align: center; color: #10a37f;'>ChromaWhisper</h1>", unsafe_allow_html=True)

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for Chat History
    with st.sidebar:
        st.markdown("<h3 style='color: #10a37f;'>Chat History</h3>", unsafe_allow_html=True)
        for i, question in enumerate(st.session_state.chat_history):
            st.markdown(f"<p>{i + 1}. {question}</p>", unsafe_allow_html=True)

    # Search Bar
    query = st.text_input("Ask a question...", key="query_input", placeholder="What is storytelling?")

    if query:
        # Add the question to chat history
        st.session_state.chat_history.append(query)

        # Display "Searching..." status
        with st.spinner("🔍 Searching..."):
            # Initialize the retriever
            retriever = Retriever()

            # Retrieve results
            results = retriever.retrieve(query)

        if results:
            # Display Results
            st.markdown("<h3 style='color: #333333;'>Response</h3>", unsafe_allow_html=True)
            for result in results:
                # Split the text into title, description, and transcript
                text_parts = result["text"].split("\n")
                title = text_parts[0].replace("Title: ", "") if text_parts[0].startswith("Title: ") else ""
                description = text_parts[1].replace("Description: ", "") if len(text_parts) > 1 and text_parts[1].startswith("Description: ") else ""
                transcript = text_parts[2].replace("Transcript: ", "") if len(text_parts) > 2 and text_parts[2].startswith("Transcript: ") else ""

                # Response Card
                st.markdown(
                    f"""
                    <div class="response-card">
                        <h4>📹 Video</h4>
                        <p><strong>Title:</strong> {title}</p>
                        <p><strong>Description:</strong> {description}</p>
                        <p><strong>Start Time:</strong> {seconds_to_min_sec(result['start_time'])}</p>
                        <p><a href="{result['video_uri']}" target="_blank">Watch Video ↗️</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Sources Section
            st.markdown(
                """
                <div class="sources-section">
                    <h4>📚 Sources</h4>
                    <p>1. <a href="https://www.youtube.com/watch?v=ftDsSB3F5kg" target="_blank">निर्देशक की भूमिका भाग - 1</a></p>
                    <p>2. <a href="https://www.youtube.com/watch?v=kKFrbhZGNNI" target="_blank">निर्देशक की भूमिका भाग - 2</a></p>
                    <p>3. <a href="https://www.youtube.com/watch?v=6qUxwZcTXHY" target="_blank">निर्देशक की भूमिका भाग - 3</a></p>
                    <p>4. <a href="https://www.youtube.com/watch?v=MspNdsh0QcM" target="_blank">स्टोरीबोर्ड का निर्माण भाग - 1</a></p>
                    <p>5. <a href="https://www.youtube.com/watch?v=Kf57KGwKa0w" target="_blank">स्टोरीबोर्ड का निर्माण भाग - 2</a></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Follow-up Questions
            st.markdown(
                """
                <div class="follow-up-questions">
                    <h4>🤔 Follow-up Questions</h4>
                    <button>What are the roles of a director?</button>
                    <button>How director selects the character and shooting location?</button>
                    <button>What is the difference between story and script?</button>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<p style='color: #666666;'>No results found.</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
