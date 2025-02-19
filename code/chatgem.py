


# this chatbot Python Flask code is to use GPT API in Azure: no datasources 
#  python C:\backupcgi\000Ed_CGI\chatbot\chatgem.py
#  http://127.0.0.1:5000/ (Press CTRL+C to quit)


'''
print(sys.executable)
C:\anaconda3\envs\graphrag-env\python.exe

conda activate graphrag-env

python --version

'''

#####Chatbot: flask + Gemini + LangChain + 
### RAG (vector DB search) + local knowledge base(creditcard.txt 
### on local drive) for RAG + context/session handling
'''
Below is an example of a complete, concise Flask chatbot 
application that uses Google Gemini (via a hypothetical 
  GeminiLLM in LangChain) together with a retrieval‐augmented
 generation (RAG) approach using a FAISS vector store built 
 from your knowledge base file (on local pc drive). In this example,
 we also maintain conversation history in a local SQLite database.
 (Make sure to install the required LangChain dependencies
  and adjust the GeminiLLM import/initialization as needed 
  for your environment.)
'''
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google.cloud import aiplatform

# --- LangChain & Google Gemini Imports ---
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ---------------------------------------------
# Load environment variables (GEMINI_API_KEY etc.)
# ---------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------
# Configuration: GCP project, model details, and local KB file path
# ---------------------------------------------
PROJECT_ID = "chatbotragproject"       # Replace with your project ID
LOCATION = "us-central1"               # e.g., us-central1
MODEL_NAME = "gemini-2.0-flash-001"      # Replace with the actual model name (or try "chat-bison" if needed)
KNOWLEDGE_FILE_PATH = r"C:\backupcgi\000Ed_CGI\creditcard.txt"  # Use raw string for Windows paths

# Initialize Google Cloud platform (ADC must be set via gcloud)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"

# ---------------------------------------------
# Load and Process the Knowledge Base
# ---------------------------------------------
if not os.path.exists(KNOWLEDGE_FILE_PATH):
    raise FileNotFoundError(f"{KNOWLEDGE_FILE_PATH} not found on local drive.")

with open(KNOWLEDGE_FILE_PATH, 'r', encoding='utf-8') as f:
    kb_content = f.read()

# Split the text into chunks (adjust chunk_size and overlap as needed)
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(kb_content)
documents = [Document(page_content=chunk) for chunk in chunks]

# Initialize embeddings and build a FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# ---------------------------------------------
# Initialize the ChatVertexAI LLM and the RAG Chain
# ---------------------------------------------
llm = ChatVertexAI(
    project_id=PROJECT_ID,
    location=LOCATION,
    model_name=MODEL_NAME,
    max_output_tokens=180,   # Adjust as needed
    temperature=0.25,
    top_p=0.95,
    api_key=GEMINI_API_KEY   # If not set, will rely on ADC
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever()
)

# ---------------------------------------------
# Conversation History Helpers (SQLite)
# ---------------------------------------------
def get_conversation_history(user_id, session_id):
    """Return chat history as a list of (user, assistant) pairs."""
    conn = sqlite3.connect('conversation_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            user_id TEXT, 
            session_id TEXT, 
            role TEXT, 
            message TEXT, 
            timestamp DATETIME
        )
    ''')
    conn.commit()
    c.execute('''
        SELECT role, message FROM conversation_history 
        WHERE user_id = ? AND session_id = ? 
        ORDER BY timestamp
    ''', (user_id, session_id))
    rows = c.fetchall()
    conn.close()
    # Convert rows into (user, assistant) pairs
    history = []
    temp_user = None
    for role, message in rows:
        if role == "user":
            temp_user = message
        elif role == "assistant" and temp_user is not None:
            history.append((temp_user, message))
            temp_user = None
    return history

def add_message_to_history(user_id, session_id, role, message):
    conn = sqlite3.connect('conversation_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            user_id TEXT, 
            session_id TEXT, 
            role TEXT, 
            message TEXT, 
            timestamp DATETIME
        )
    ''')
    c.execute('''
        INSERT INTO conversation_history (user_id, session_id, role, message, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, session_id, role, message, datetime.utcnow()))
    conn.commit()
    conn.close()

def reset_history_internal():
    try:
        conn = sqlite3.connect('conversation_history.db')
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS conversation_history')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Reset history error:", e)
        return False

# ---------------------------------------------
# Flask Routes
# ---------------------------------------------
@app.route('/')
def chat():
    # Optionally reset conversation history at the start of a session
    reset_history_internal()
    return render_template('chat.html')

@app.route('/reset_history', methods=['GET'])
def reset_history():
    reset_history_internal()
    return render_template('chat.html')

@app.route('/send', methods=['POST'])
def send_message():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': 'No message received.'})
    # For demonstration, we use fixed user and session IDs.
    user_id = 'unique_user_id'
    session_id = 'unique_session_id'
    
    # Retrieve conversation history as (user, assistant) pairs
    chat_history = get_conversation_history(user_id, session_id)
    
    # Run the retrieval-augmented generation chain
    result = qa_chain({"question": user_message, "chat_history": chat_history})
    answer = result.get("answer", "I'm sorry, I couldn't generate an answer.")
    
    # Save the new exchange in conversation history
    add_message_to_history(user_id, session_id, "user", user_message)
    add_message_to_history(user_id, session_id, "assistant", answer)
    
    return jsonify({'response': answer})

@app.route('/dialog_hist')
def dialog_hist():
    # Return the full conversation history as JSON (for debugging/inspection)
    user_id = 'unique_user_id'
    session_id = 'unique_session_id'
    history = get_conversation_history(user_id, session_id)
    return jsonify(history)

# ---------------------------------------------
# Main Entry Point
# ---------------------------------------------
if __name__ == '__main__':
    app.run()
