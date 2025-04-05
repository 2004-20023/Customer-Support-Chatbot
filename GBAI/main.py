import os
import json
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import speech_recognition as sr
from gtts import gTTS
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.globals import set_verbose
from langdetect import detect
from deep_translator import GoogleTranslator
from werkzeug.security import generate_password_hash, check_password_hash

# ✅ File Paths for Storing User and Chat Data
USERS_FILE = "users.json"
CHATS_FILE = "chat_history.json"

# ✅ Ensure Directories Exist
current_dir = os.path.dirname(os.path.abspath(__file__))
documents_dir = os.path.join(current_dir, "data")
persist_directory = os.path.join(documents_dir, "db", "faiss_db")

os.makedirs(persist_directory, exist_ok=True)
os.makedirs(documents_dir, exist_ok=True)

# ✅ Load Data Functions
def load_data(file_path, default_data={}):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return default_data

def save_data(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

users_db = load_data(USERS_FILE)
chats_db = load_data(CHATS_FILE)

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Logging Setup
logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Define Company-Related Keywords & Greetings
COMPANY_KEYWORDS = {"goodbooks", "goodbooks technologies"}
EMPLOYEE_KEYWORDS = {"employee", "details", "info", "find", "get", "id", "name", "search", "about", "who"}
GREETINGS = {"hi", "hello", "hey", "good morning", "good evening"}

# ✅ Load Documents
all_docs = []
txt_files = [f for f in os.listdir(documents_dir) if f.endswith(".txt")]

if txt_files:
    for file_name in txt_files:
        try:
            file_path_full = os.path.join(documents_dir, file_name)
            text_loader = TextLoader(file_path_full)
            docs = text_loader.load()
            all_docs.extend(docs)
        except Exception as e:
            logging.error(f"❌ Error loading {file_name}: {e}")

# ✅ Split Documents
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=2)
text_chunks = splitter.split_documents(all_docs) if all_docs else []

# ✅ Initialize Embeddings & Vector Store
embeddings = OllamaEmbeddings(model="nomic-embed-text")

if text_chunks:
    try:
        vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    except:
        vectorstore = FAISS.from_documents(text_chunks, embeddings)
        vectorstore.save_local(persist_directory)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
else:
    retriever = None

# ✅ Initialize LLM & QA Chain
llm = ChatOllama(model="mistral")
set_verbose(True)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff") if retriever else None

# ✅ Multi-Language Support
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_text(text, src_lang, target_lang):
    if src_lang != target_lang:
        return GoogleTranslator(source=src_lang, target=target_lang).translate(text)
    return text

# ✅ Authentication Models
class User(BaseModel):
    username: str
    password: str

class Message(BaseModel):
    content: str
    username: str

@app.post("/signup")
async def signup(user: User):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists!")
    users_db[user.username] = generate_password_hash(user.password)
    save_data(USERS_FILE, users_db)
    return {"message": "✅ Account created successfully!"}

@app.post("/login")
async def login(user: User):
    if user.username in users_db and check_password_hash(users_db[user.username], user.password):
        return {"message": "✅ Logged in successfully!"}
    raise HTTPException(status_code=401, detail="❌ Invalid credentials")

# ✅ Chatbot API
@app.post("/api/chat")
async def chat(message: Message):
    """Handles chat queries."""
    user_input = message.content.strip()
    input_lang = detect_language(user_input)
    query_in_english = translate_text(user_input, input_lang, "en")

    if query_in_english.lower() in GREETINGS:
        response = "Hello! How can I assist you today?"
    elif any(keyword in query_in_english.lower() for keyword in COMPANY_KEYWORDS):
        response = qa.run(query_in_english) if qa else "I couldn't find information about GoodBooks."
    elif any(keyword in query_in_english.lower() for keyword in EMPLOYEE_KEYWORDS):
        response= qa.run(query_in_english) if qa else "Sorry , Employee details not found!"
    else:
        response = llm.invoke(query_in_english).content if llm else "I'm sorry, I don't have enough information."

    translated_response = translate_text(response, "en", input_lang)

    # ✅ Store chat history
    user_chats = chats_db.get(message.username, [])
    user_chats.append({"message": user_input, "response": translated_response})
    chats_db[message.username] = user_chats
    save_data(CHATS_FILE, chats_db)

    return {"response": translated_response}

# ✅ Retrieve Chat History
@app.get("/chat-history/{username}")
async def get_chat_history(username: str):
    return {"history": chats_db.get(username, [])}

# ✅ Voice Input Processing
@app.post("/voice-input")
async def capture_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return {"transcribed_text": text}
        except sr.UnknownValueError:
            return {"error": "Sorry, I couldn't understand you."}
        except sr.RequestError:
            return {"error": "Could not request results, check your internet connection."}

# ✅ Text-to-Speech Response
@app.post("/speak")
async def speak_response(response: Message):
    tts = gTTS(text=response.content, lang="en")
    audio_file = "response.mp3"
    tts.save(audio_file)
    return {"audio_file": audio_file}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


