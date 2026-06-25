# Local RAG Sales Assistant 🚗

This project is a fully local, privacy-first **RAG (Retrieval-Augmented Generation)** assistant designed for the automotive sales industry. It processes various document types (CSV, JSON, TXT) and answers user queries using a local LLM without any data leaving your machine.

### 🌟 Features
* **100% Privacy:** All documents and queries are processed locally. No external APIs are used for generation.
* **Multi-Format Ingestion:** Can read and understand Excel/CSV pricing lists, JSON car catalogs, and TXT/Word policy documents.
* **Multilingual Embedding:** Uses HuggingFace's multilingual models to accurately map non-English queries to the correct documents.
* **Source Tracking:** Provides the exact source document name for every answer it generates to prevent hallucination.

### 🛠️ Tech Stack
* **LLM:** [Ollama](https://ollama.com/) (Llama-3 8B)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`)

### 🚀 Quick Start

# 1. Clone the repository:**
``bash
git clone [https://github.com/sesliseher/local-rag-sales-assistant.git](https://github.com/sesliseher/local-rag-sales-assistant.git)
cd local-rag-sales-assistant
# 2. Create and activate a virtual environment:
python -m venv venv

 For Windows:
.\venv\Scripts\activate

For Mac/Linux:
source venv/bin/activate

# 3. Install dependencies:
pip install -r requirements.txt

# 4. Start Ollama and download the model:
Make sure Ollama is installed on your system. Open a separate terminal and run:
ollama run llama3


# 5. Run the application:
python app.py

Final! You are create a local RAG 🚀👏
