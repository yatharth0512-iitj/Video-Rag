# WHAT IS RAG ? 
What is RAG (Retrieval-Augmented Generation)?
Retrieval-Augmented Generation (RAG) is a technique that enhances the performance of language models by combining two powerful components:

Retriever: A model or process that searches a large database (e.g., a corpus of documents or knowledge base) for relevant information based on a given query.
Generator: A model, typically a language model like GPT or BERT, that generates responses or outputs based on the retrieved information.
RAG allows the generation of answers or content by leveraging external knowledge (retrieved documents) to improve the quality and accuracy of the generated output. It is particularly useful in situations where the model's internal knowledge might be limited or outdated, or when the task requires highly specific information that would be hard for a standalone model to memorize.

### How it Works:

**Step 1: Query and Retrieve:**  
Given a user query, the retriever searches a knowledge base (like a vector database) to find the most relevant documents or information.  

**Step 2: Generate Output:**  
The generator uses the retrieved information to generate a more informed response, incorporating details from the knowledge base.  

RAG is commonly used in tasks like question answering, document generation, summarization, and more, as it allows the model to work with external information in real time.

# Video Metadata Processing and Retrieval Pipeline

This pipeline efficiently processes video metadata by chunking, embedding, storing, and retrieving relevant information using state-of-the-art tools. It is implemented through the following modular components:

## **Modular Components**

### **1. MetadataChunker**
- **Description**: Splits video metadata into smaller, meaningful chunks.
- **Each chunk consists of**:
  - **Title** and **Description** of the video.
  - **Transcript segments** (start time, end time, text).
  - **Frame descriptions**, if available.
  - **Video URI** to reference the original video.

### **2. EmbeddingGenerator**
- **Description**: Uses the `SentenceTransformer` model (`all-MiniLM-L6-v2`) to generate text embeddings.
- **Task**: Converts structured chunks into a format suitable for vector storage.
- **Functionality**: Combines transcript, title, and frame descriptions for embedding.

### **3. VectorDB**
- **Description**: Stores the generated embeddings and metadata in a vector database (`ChromaDB`).
- **Features**:
  - Indexes embeddings for efficient querying.
  - Stores metadata like start time, end time, and video URI for context retrieval.

### **4. Retriever**
- **Description**: Retrieves the most relevant video chunks based on a user query.
- **How it works**:
  - Converts the query into an embedding.
  - Searches for similar embeddings in the `ChromaDB`.
  - Returns relevant transcript sections along with video details.

### **5. Process and Query Functions**
- **`process_all_metadata_for_vectordb(metadata_dir)`**
  - Loads all metadata JSON files from the given directory.
  - Uses `MetadataChunker` to generate chunks.
  - Computes embeddings and stores them in the vector database.
  
- **`query_vectordb(query)`**
  - Retrieves relevant video transcript sections based on the given query.
  - Displays video URIs and timestamps for easy reference.

---

## **How to Run**

Follow the steps below to run the pipeline:

### **1. Set Up the Environment**

First, create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### **2. Install Dependencies**

Install the required dependencies using the `requirements.txt` file. Create the `requirements.txt` with the following contents:

```bash
pip install -r requirements.txt
```

To run the Streamlit application, execute:

```bash
streamlit run app.py
```
