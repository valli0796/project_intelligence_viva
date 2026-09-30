import streamlit as st
import requests
import fitz
import chromadb
import hashlib
import secrets
import json
import os
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="VivaIQ", page_icon="🎓", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
HISTORY_FILE = os.path.join(BASE_DIR, "viva_history.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_chroma():
    return chromadb.PersistentClient(path=CHROMA_DIR)

embedding_model = load_embedding_model()
chroma_client = load_chroma()

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return default
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
    return salt, hashed

def verify_password(password, salt, stored_hash):
    _, hashed = hash_password(password, salt)
    return secrets.compare_digest(hashed, stored_hash)

def register_user(username, password):
    users = load_json(USERS_FILE, {})
    if username in users:
        return False, "Username already exists."
    salt, hashed = hash_password(password)
    users[username] = {"salt": salt, "password": hashed}
    save_json(USERS_FILE, users)
    return True, "Registration successful."

def login_user(username, password):
    users = load_json(USERS_FILE, {})
    user = users.get(username)
    if not user:
        return False
    return verify_password(password, user["salt"], user["password"])

def save_history(username, activity, details):
    history = load_json(HISTORY_FILE, [])
    history.append({
        "username": username,
        "activity": activity,
        "details": details,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json(HISTORY_FILE, history)

def get_user_collection(username):
    collection_name = "user_" + hashlib.sha256(username.encode()).hexdigest()[:24]
    return chroma_client.get_or_create_collection(name=collection_name)

def extract_pdf_text(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as document:
        for page in document:
            text += page.get_text() + "\n"
    return text.strip()

def split_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def index_document(username, text, filename):
    collection = get_user_collection(username)
    chunks = split_text(text)
    if not chunks:
        return 0
    document_id = hashlib.sha256((filename + text[:200]).encode()).hexdigest()[:16]
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{document_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk": i} for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    return len(chunks)

def retrieve_context(username, question, top_k=3):
    collection = get_user_collection(username)
    if collection.count() == 0:
        return ""
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )
    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents)

def ask_ollama(prompt, system_prompt="You are VivaIQ, an academic viva preparation assistant. Give clear and accurate answers."):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "No response received.")
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Please start Ollama and check that the model is installed."
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again."
    except requests.exceptions.RequestException as error:
        return f"Error communicating with Ollama: {error}"

def generate_questions(text, count=5):
    prompt = f"Generate {count} important project viva questions based on the following project content. Return only a numbered list of questions.\n\nProject content:\n{text[:6000]}"
    return ask_ollama(prompt)

def evaluate_answer(question, answer, context=""):
    prompt = f"""Evaluate this student's viva answer.
Question: {question}
Student's answer: {answer}
Reference information: {context[:2500]}
Give a score out of 10, explain what was correct, identify missing points, and provide an improved sample answer. Start with SCORE: followed by a number from 0 to 10."""
    feedback = ask_ollama(prompt)
    match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", feedback, re.IGNORECASE)
    score = float(match.group(1)) if match else 0
    score = min(10, max(0, score))
    return score, feedback

def show_login():
    st.title("🎓 VivaIQ")
    st.subheader("AI-Powered Project Viva Preparation")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if login_user(username.strip(), password):
                    st.session_state["username"] = username.strip()
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username", key="register_username")
            new_password = st.text_input("Choose a password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                if not new_username.strip() or not new_password:
                    st.error("Please fill in all fields.")
                elif len(new_password) < 6:
                    st.error("Password must contain at least 6 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, message = register_user(new_username.strip(), new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

def show_dashboard():
    username = st.session_state["username"]
    st.title("🎓 VivaIQ Dashboard")
    st.write(f"Welcome, **{username}**!")

    with st.sidebar:
        st.title("VivaIQ Menu")
        page = st.radio(
            "Navigate",
            ["Dashboard", "Upload Project PDF", "AI Chatbot", "Mock Viva", "Reports & History", "About"]
        )
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if page == "Dashboard":
        st.header("Your Project Viva Assistant")
        col1, col2, col3 = st.columns(3)
        history = load_json(HISTORY_FILE, [])
        user_history = [item for item in history if item["username"] == username]
        col1.metric("Total Activities", len(user_history))
        col2.metric("PDF Documents", len(list_collection_documents(username)))
        col3.metric("Mock Viva Sessions", sum(1 for item in user_history if item["activity"] == "Mock Viva"))
        st.info("Upload your project PDF to start preparing for your viva.")
        st.markdown("### Features")
        st.markdown("- Upload and understand project documents")
        st.markdown("- Ask questions using the AI chatbot")
        st.markdown("- Practice mock viva questions")
        st.markdown("- Receive answer scores and feedback")
        st.markdown("- Review your previous activities")

    elif page == "Upload Project PDF":
        st.header("Upload Your Project PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file is not None:
            if st.button("Process PDF", use_container_width=True):
                try:
                    text = extract_pdf_text(uploaded_file)
                    if not text:
                        st.error("No readable text found in this PDF.")
                    else:
                        chunks = index_document(username, text, uploaded_file.name)
                        saved_path = os.path.join(UPLOAD_DIR, f"{username}_{uploaded_file.name}")
                        with open(saved_path, "wb") as file:
                            file.write(uploaded_file.getvalue())
                        st.session_state["project_text"] = text
                        st.session_state["project_filename"] = uploaded_file.name
                        save_history(username, "PDF Upload", uploaded_file.name)
                        st.success(f"PDF processed successfully. Created {chunks} text chunks.")
                        with st.expander("Preview extracted text"):
                            st.text_area("Extracted content", text[:5000], height=300)
                except Exception as error:
                    st.error(f"Unable to process PDF: {error}")

    elif page == "AI Chatbot":
        st.header("Ask Your Project Questions")
        st.caption("Ask questions about your uploaded project or general viva topics.")
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []

        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about your project...")
        if question:
            st.session_state["chat_messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            context = retrieve_context(username, question)
            if context:
                prompt = f"Answer the user's question using the project context below. If the answer is not in the context, say so and then provide general guidance if useful.\n\nProject context:\n{context}\n\nQuestion: {question}"
            else:
                prompt = question

            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    answer = ask_ollama(prompt)
                    st.markdown(answer)

            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
            save_history(username, "Chatbot", question)

        if st.button("Clear Chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

    elif page == "Mock Viva":
        st.header("Mock Viva Practice")
        st.write("Practice answering questions about your project.")
        collection = get_user_collection(username)
        if collection.count() == 0:
            st.warning("Please upload your project PDF before starting a project-based mock viva.")
        else:
            if "viva_questions" not in st.session_state:
                st.session_state["viva_questions"] = []
            if "viva_answers" not in st.session_state:
                st.session_state["viva_answers"] = {}
            if "viva_feedback" not in st.session_state:
                st.session_state["viva_feedback"] = {}

            number_of_questions = st.selectbox("Number of questions", [3, 5, 10], index=1)

            if st.button("Generate Viva Questions", use_container_width=True):
                context = retrieve_context(username, "Project overview, objectives, methodology, technologies, results and limitations", top_k=5)
                with st.spinner("Generating questions..."):
                    generated = generate_questions(context, number_of_questions)
                questions = []
                for line in generated.splitlines():
                    cleaned = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", line).strip()
                    if cleaned:
                        questions.append(cleaned)
                st.session_state["viva_questions"] = questions[:number_of_questions]
                st.session_state["viva_answers"] = {}
                st.session_state["viva_feedback"] = {}

            questions = st.session_state["viva_questions"]
            if questions:
                st.subheader("Answer the Questions")
                for index, question in enumerate(questions):
                    st.markdown(f"**Question {index + 1}: {question}**")
                    st.session_state["viva_answers"][str(index)] = st.text_area(
                        "Your answer",
                        value=st.session_state["viva_answers"].get(str(index), ""),
                        key=f"answer_{index}",
                        height=100
                    )

                if st.button("Submit Viva Answers", use_container_width=True):
                    total_score = 0
                    feedback_results = []
                    with st.spinner("Evaluating your answers..."):
                        for index, question in enumerate(questions):
                            answer = st.session_state["viva_answers"].get(str(index), "")
                            context = retrieve_context(username, question, top_k=2)
                            score, feedback = evaluate_answer(question, answer, context)
                            total_score += score
                            feedback_results.append({
                                "question": question,
                                "answer": answer,
                                "score": score,
                                "feedback": feedback
                            })

                    average_score = total_score / len(questions) if questions else 0
                    st.session_state["viva_feedback"] = feedback_results
                    st.session_state["viva_average_score"] = average_score
                    save_history(username, "Mock Viva", {
                        "questions": len(questions),
                        "average_score": round(average_score, 2)
                    })

                if st.session_state["viva_feedback"]:
                    st.divider()
                    st.subheader("Viva Results")
                    st.metric("Average Score", f"{st.session_state['viva_average_score']:.1f}/10")
                    for index, result in enumerate(st.session_state["viva_feedback"]):
                        with st.expander(f"Question {index + 1} — Score: {result['score']:.1f}/10"):
                            st.write("**Question:**", result["question"])
                            st.write("**Your Answer:**", result["answer"] or "No answer provided.")
                            st.markdown(result["feedback"])

    elif page == "Reports & History":
        st.header("Reports and Activity History")
        history = load_json(HISTORY_FILE, [])
        user_history = [item for item in history if item["username"] == username]
        if not user_history:
            st.info("No activity history available yet.")
        else:
            st.subheader("Activity Summary")
            st.metric("Total Activities", len(user_history))
            for item in reversed(user_history):
                with st.expander(f"{item['activity']} — {item['timestamp']}"):
                    st.write(item["details"])
            report = {
                "username": username,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "activities": user_history
            }
            st.download_button(
                "Download Activity Report",
                data=json.dumps(report, indent=4),
                file_name=f"{username}_viva_report.json",
                mime="application/json",
                use_container_width=True
            )

    elif page == "About":
        st.header("About VivaIQ")
        st.write("VivaIQ is an AI-powered project viva preparation platform.")
        st.markdown("### Technologies Used")
        st.markdown("- Python")
        st.markdown("- Streamlit")
        st.markdown("- Ollama with Llama 3.2")
        st.markdown("- ChromaDB")
        st.markdown("- Sentence Transformers")
        st.markdown("- PyMuPDF")
        st.markdown("- JSON-based user accounts and history")
        st.markdown("### Main Features")
        st.markdown("- User login and registration")
        st.markdown("- PDF text extraction and semantic retrieval")
        st.markdown("- AI-powered project chatbot")
        st.markdown("- Mock viva questions and answer evaluation")
        st.markdown("- Activity history and downloadable reports")

def list_collection_documents(username):
    collection = get_user_collection(username)
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    filenames = set()
    for metadata in results.get("metadatas", []):
        if metadata and "filename" in metadata:
            filenames.add(metadata["filename"])
    return sorted(filenames)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    show_dashboard()
else:
    show_login()