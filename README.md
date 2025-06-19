# Video Transcript Chatbot

A beginner-friendly Gradio app that turns any YouTube video into a conversational chatbot using LangChain and Hugging Face Inference API.

---

## Features

- **Dynamic Video Input**: Paste a full YouTube URL or raw video ID.  
- **Embedding Model Selection**: Pick any HF embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`).  
- **LLM Model Selection**: Choose any HF text-generation model (default: `meta-llama/Llama-3.1-8B-Instruct`).  
- **Secure Token Entry**: You must enter your own HF API token at runtime—no hard-coded defaults.  
- **Conversational Memory**: Multi-turn chat history is preserved.  
- **Retrieval-Augmented Generation**: Uses FAISS + transcript context to ground answers.

---

## Prerequisites

- **Python 3.8+**  
- **Hugging Face API Token** with Inference access:  
  https://huggingface.co/settings/tokens  
- **Git** (for cloning the repo)

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/yt-rag-chatbot.git
   cd yt-rag-chatbot
2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
3. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows

**Usage**
1. **Start the app:**
   ```bash
   python app.py
2. **Open** your browser at the local URL (e.g. http://127.0.0.1:7860)
3. **Use the UI:**

- **YouTube Video URL or ID:** Paste your link/ID.

- **Embedding Model:** Leave default or enter another HF embedding model.

- **LLM Model:** Enter your desired HF LLM repo.

- **Your HF API Token:** Paste your token (input hidden).

- Click **Initialize Chat** to load and index the transcript.

- Ask questions in the chat window to interact with the video content.

  
**Customization**

- **Default Models:** Edit the default values for embedding_model_input and llm_model_input in app.py.

- **Retrieval Size:** Change the k value in the retriever configuration:
  ```python
  retriever = vector_store.as_retriever(search_kwargs={'k': 4})
 


   
