import os, re, uuid
from flask import Flask, render_template, request, session, jsonify
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

# ---- helpers ---------------------------------------------------------------

def extract_video_id(url_or_id: str) -> str:
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url_or_id)
    return m.group(1) if m else url_or_id.strip()

def build_chain(video_id, token, embed_model, llm_model):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token.strip()

    try:
        parts = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(p["text"] for p in parts)
    except TranscriptsDisabled:
        transcript = ""

    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs       = splitter.create_documents([transcript])
    embeddings = HuggingFaceEndpointEmbeddings(model=embed_model)

    vectordb   = FAISS.from_documents(docs, embeddings)
    retriever  = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_tmpl = """
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

{context}
Question: {question}
"""
    prompt   = PromptTemplate(template=prompt_tmpl,
                              input_variables=["context", "question"])
    memory   = ConversationBufferMemory(memory_key="chat_history",
                                        return_messages=True)

    hf_llm   = HuggingFaceEndpoint(
        repo_id=llm_model,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.2
    )

    chat_model = ChatHuggingFace(llm=hf_llm, verbose=False)

    return ConversationalRetrievalChain.from_llm(
        llm      = chat_model,
        retriever=retriever,
        memory   = memory,
        chain_type="stuff",
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

# Keep chains in RAM keyed by a short session-specific ID.
CHAINS = {}

def get_chain():
    chain_id = session.get("chain_id")
    return CHAINS.get(chain_id)

# ---- routes ----------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    # If user has no chain yet, generate a session ID now
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return render_template("index.html")

@app.route("/setup", methods=["POST"])
def setup():
    data          = request.json
    video_input   = data["video"]
    embed_model   = data.get("embed_model") or "sentence-transformers/all-MiniLM-L6-v2"
    llm_model     = data.get("llm_model")   or "meta-llama/Llama-3.1-8B-Instruct"
    hf_token      = data["hf_token"]

    video_id      = extract_video_id(video_input)
    chain         = build_chain(video_id, hf_token, embed_model, llm_model)

    chain_id           = uuid.uuid4().hex
    CHAINS[chain_id]   = chain
    session["chain_id"] = chain_id

    return jsonify({"status": "initialized"})

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]
    chain    = get_chain()
    if chain is None:
        return jsonify({"error": "Chain not initialized"}), 400

    chat_history = chain.memory.chat_memory.messages  # already stored internally
    result       = chain({"question": user_msg, "chat_history": chat_history})
    answer       = result.get("answer") or result.get("result", "")
    return jsonify({"reply": answer})

# ---- run -------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
