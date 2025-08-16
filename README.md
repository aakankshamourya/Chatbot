

<img width="1280" height="720" alt="AgenticAI Chatbot" src="https://github.com/user-attachments/assets/430cf82d-4e26-430a-8dfe-256f5ddf7c7d" />





















Developed advanced contextual and knowledge-based chatbot solutions using GenAi LLM, effectively handling both unstructured and structured data. Integrated GenAi with Langchain to manage workflow orchestration, while interacting with a vast 10 GB of unstructured data. Conducted extensive prompt engineering & RAG techniques to optimize the system’s responsiveness and intelligence & deployed GenAi solutions on the client’s VM.& Implemented the streaming functionality like Chat GPT interface in solutions . Which helps the internal audience/employees to understand the Research data much better.


Objective: interact with the research data of Almarai , and find the relevant information quickly and accurately 






pipeline:

1: unstactured data

LangChain(memory, agent, tool,chain, vdb, prompt, LLM)


2: embedding (openai embedding)

    chunk size

3: vector database(faiss)/facebook 

   RAG Techniques(Knowledge base + prompt + user question)


Knowledge graph(vector search + RAG)


Similarity search (threshold level) to get the best accurate result

4: LLM model

5: prompte engineering


  1: greeting (hi ,hello),

6: relevance(top and latest result)

   verbosity (summary length)

7: citation generation and sorting latest to old

8: streaming capabilities

9: JSON output with HTML tags

10: endpoint(Rest api/flask): Request functionality like download, upload, delete

11: logging capabilities has been monitored at every stage(CI/CD)



Code automation with CI/CD pipeline 


# Chatbot – Contextual & Knowledge-Based AI Assistant


## Overview

This chatbot leverages Generative AI (LLM) and Retrieval-Augmented Generation (RAG) techniques to process both unstructured and structured research data. It integrates LangChain for orchestration, FAISS for efficient vector search, and OpenAI embeddings to deliver accurate, context-aware responses.

Key capabilities include:
- Handling large volumes of unstructured data (~10 GB)
- Advanced prompt engineering and RAG for intelligent querying
- Streaming-like interfaces similar to ChatGPT
- Well-structured JSON + HTML responses
- Complete logging and CI/CD-enabled deployment

---

## System Architecture

Below is a simplified architecture diagram showing how data flows through the system:

```

Unstructured Data (10 GB)
↓
Chunking → Embedding (OpenAI) → Vector DB (FAISS)
↓
Similar Query Retrieval (Threshold-based)
↓
RAG (Knowledge Base + Prompt + Query)
↓
Knowledge Graph + Vector Search
↓
LLM (Generative AI) with Prompt Engineering
↓
Response Formatting (Summary, Citations, JSON + HTML, Streaming)
↓
Flask REST API Endpoint with Logging & CI/CD

````

---

## Workflow Flowchart

```mermaid
flowchart TD
    A[User Query] --> B[Vector Search in FAISS]
    B --> C{Similarity Threshold Reached?}
    C -- Yes --> D[Retrieve Context → Perform RAG]
    C -- No --> E[Return “No relevant data found”]
    D --> F[Generate Response via LLM (Prompt Engineering)]
    F --> G[Format Output (Summary + Citations, JSON/HTML)]
    G --> H[Stream Response to UI]
    H --> I[Log Interaction via CI/CD Pipeline]
````

---

## Pipeline & Code Explanation

### 1. **Data Ingestion & Embedding**

* Load and chunk large research documents (e.g., PDFs, text files).
* Use OpenAI embeddings to encode text into vectors.
* Insert embeddings into a FAISS vector database for fast similarity lookup.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Example:
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(chunks, embeddings)
```

### 2. **Similarity Search & RAG**

* On receiving a query, compute its embedding.
* Search FAISS for nearest vector(s) using a similarity threshold.
* Retrieve top results and pass context + the query to the LLM via RAG.

```python
query_emb = embeddings.embed_query(user_query)
docs = vector_db.similarity_search_by_vector(query_emb, k=5)
```

### 3. **Knowledge Graph Integration**

* Optional: Load knowledge graph (e.g., Neo4j) with relevant metadata or entities.
* Use it to enrich context or resolve relationships through LangChain’s Graph Agent.

```python
# Example using LangChain Graph Agent
from langchain.agents import GraphChainAgent
# Setup and run graph agent with user query
```

### 4. **LLM Response & Prompt Engineering**

* Design prompts that:

  * Provide context from retrieved chunks
  * Enforce structure (e.g., JSON with citations, summary length limit)
* Support streaming responses for gradual rendering.

```python
prompt = f"""
Use the following context to answer:
{retrieved_context}

Provide a concise summary, include citation, and return output in JSON with HTML tags.
"""
response = llm.generate(prompt, stream=True)
```

### 5. **Formatting & Output**

* Package the response into JSON, embed HTML as needed (e.g., for rich display in UI).
* Ensure citation order (latest information first) and summary brevity.

```json
{
  "summary": "<p>...</p>",
  "citations": ["source1", "source2"],
  "html": "<div>...</div>"
}
```

### 6. **API Endpoint & Logging**

* Deploy via Flask REST API endpoints (upload, download, delete).
* Implement comprehensive logging for each stage and integrate with CI/CD workflows.

```python
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(...)
# Define your endpoints with handlers and logging here
```

---

## API & Logging Features

* **RESTful Endpoints**:

  * `POST /query` – for submitting user queries and receiving responses
  * `POST /upload`, `DELETE /delete` – manage data ingestion

* **CI/CD & Logging**:

  * Logs capture query inputs, similarity scores, retrieved context, prompt used, LLM output, formatting, and timestamps for traceability and monitoring.

---

## Setup & Usage

1. **Clone the Repository**

```bash
git clone https://github.com/aakankshamourya/Chatbot.git
cd Chatbot
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Setup Vector DB**

```bash
python scripts/prepare_embeddings.py
```

4. **Run the API**

```bash
python app.py
```

5. **Query the Bot**

Send a POST request:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the method used in Excel data cleaning"}'
```

---

## Summary

This Chatbot intelligently interacts with research data using a combination of:

* LangChain for orchestration
* FAISS-powered vector retrieval
* RAG-enhanced responses
* Flexible formatting with citations and streaming
* API-driven architecture integrated with CI/CD



