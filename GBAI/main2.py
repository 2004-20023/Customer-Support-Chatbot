import os
import json
import logging
from fastapi import FastAPI, HTTPException
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

# ✅ File Paths
USERS_FILE = "users.json"
CHATS_FILE = "chat_history.json"
EMPLOYEES_FILE = "employees.json"
COMPANY_DETAILS_FILE = "data/db/company_details.txt"

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
employee_data = load_data(EMPLOYEES_FILE)

# ✅ Load Company Details
if os.path.exists(COMPANY_DETAILS_FILE):
    loader = TextLoader(COMPANY_DETAILS_FILE)
    company_docs = loader.load()
else:
    company_docs = []

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

# ✅ Define Keywords & Greetings
COMPANY_KEYWORDS = {"goodbooks", "goodbooks technologies"}
GREETINGS = {"hi", "hello", "hey", "good morning", "good evening"}


# ✅ Load Employee Data into FAISS
def prepare_employee_vectors(employee_data):
    employee_texts = []
    for emp_id, details in employee_data.items():
        text = f"ID: {emp_id}, Name: {details['name']}, Role: {details['role']}, Salary: {details['salary']}, Department: {details['department']}"
        employee_texts.append(text)

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=2)
    employee_chunks = splitter.create_documents(employee_texts)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(employee_chunks, embeddings)
    vectorstore.save_local(persist_directory)
    return vectorstore


# ✅ Initialize FAISS for Employee Search
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = prepare_employee_vectors(employee_data) if employee_data else None
retriever = vectorstore.as_retriever(search_kwargs={'k': 1}) if vectorstore else None

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


# ✅ Signup & Login
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
    user_input = message.content.strip()
    input_lang = detect_language(user_input)
    query_in_english = translate_text(user_input, input_lang, "en")

    if query_in_english.lower() in GREETINGS:
        response = "Hello! How can I assist you today?"
    elif any(keyword in query_in_english.lower() for keyword in COMPANY_KEYWORDS):
        response = qa.run(query_in_english) if qa else "I couldn't find information about GoodBooks."
    elif query_in_english.lower().startswith("employee details"):
        response = qa.run(query_in_english) if qa else "Employee details not found!"
    else:
        response = llm.invoke(query_in_english).content if llm else "I'm sorry, I don't have enough information."

    translated_response = translate_text(response, "en", input_lang)
    user_chats = chats_db.get(message.username, [])
    user_chats.append({"message": user_input, "response": translated_response})
    chats_db[message.username] = user_chats
    save_data(CHATS_FILE, chats_db)
    return {"response": translated_response}


# ✅ Retrieve Chat History
@app.get("/chat-history/{username}")
async def get_chat_history(username: str):
    return {"history": chats_db.get(username, [])}


# ✅ Run FastAPI Server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)