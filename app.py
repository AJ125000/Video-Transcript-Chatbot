import os
import re
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# No default token: user must supply their Hugging Face API token via the UI

def extract_video_id(url_or_id: str) -> str:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url_or_id)
    return match.group(1) if match else url_or_id

# Load, embed, and index the transcript
def load_vector_store(video_id: str, huggingface_token: str, embedding_model: str):
    # Temporarily set the token for embedding calls
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_token.strip()
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = ' '.join(chunk['text'] for chunk in transcript_list)
    except TranscriptsDisabled:
        transcript = ''
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])
    embeddings = HuggingFaceEndpointEmbeddings(
        model=embedding_model,
        huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN']
    )
    return FAISS.from_documents(docs, embeddings)

# Initialize/reinitialize the QA chain
def setup(video_input, embedding_model, llm_model, huggingface_token):
    video_id = extract_video_id(video_input)
    vector_store = load_vector_store(video_id, huggingface_token, embedding_model)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    prompt_template = '''
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

{context}
Question: {question}
'''
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Configure the LLM endpoint
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_token.strip()
    hf_llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        task='text-generation',
        max_new_tokens=512,
        temperature=0.2,
        huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN']
    )
    chat_model = ChatHuggingFace(llm=hf_llm, verbose=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        memory=memory,
        chain_type='stuff',
        return_source_documents=False
    )

    # Reset chat history
    return [], [], qa_chain

# Handle chat interactions
def respond(message, chat_history, qa_chain):
    result = qa_chain({'question': message, 'chat_history': chat_history})
    answer = result.get('answer') or result.get('result')
    chat_history.append((message, answer))
    return chat_history, chat_history

# Gradio UI layout
with gr.Blocks() as demo:
    gr.Markdown('# Video Transcript Chatbot')
    with gr.Row():
        video_input = gr.Textbox(label='YouTube Video URL or ID', value='')
        embedding_model_input = gr.Textbox(
            label='Embedding Model (default: sentence-transformers/all-MiniLM-L6-v2)',
            value='sentence-transformers/all-MiniLM-L6-v2'
        )
        llm_model_input = gr.Textbox(label='LLM Model Repo (e.g. google/flan-t5-large)', value='meta-llama/Llama-3.1-8B-Instruct')
        token_input = gr.Textbox(label='Your HF API Token', placeholder='hf_...', type='password')
        init_btn = gr.Button('Initialize Chat')

    chatbot = gr.Chatbot()
    chat_state = gr.State([])
    chain_state = gr.State(None)

    init_btn.click(
        setup,
        inputs=[video_input, embedding_model_input, llm_model_input, token_input],
        outputs=[chatbot, chat_state, chain_state]
    )

    txt = gr.Textbox(placeholder='Ask a question about the video...', show_label=False)
    txt.submit(respond, inputs=[txt, chat_state, chain_state], outputs=[chatbot, chat_state])

    gr.Button('Clear Chat').click(lambda: ([], []), None, [chatbot, chat_state])

if __name__ == '__main__':
    demo.launch()  # pass share=True or host/port if needed
