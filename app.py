import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pydub import AudioSegment
import speech_recognition as sr
import random
import os
import tempfile

# === Configuration ===
genai.configure(api_key="AIzaSyAc_cTPTtI7QmG08w99fKKingFKuHRl9q8")
pc = Pinecone(api_key="pcsk_4twyfY_DwDPWiJL826wcfZREc6wbd8cNpTkw6ocLTML4g19gMxd57yyLgNGPD18WDFkGQ1")
index = pc.Index("java")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

java_questions = [
    "What is Java", "What is JVM", "What is method overloading", 
    "What is polymorphism", "What is serialization", 
    "What is shallow and deep copy", "What is File class in Java"
]

def transcribe_audio(file):
    recognizer = sr.Recognizer()
    try:
        audio = AudioSegment.from_file(file)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format="wav")

        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        return f"[Error] {e}"

# === Streamlit UI ===
st.set_page_config(page_title="AI Mock Interview", layout="centered")
st.title("💼 AI Mock Interview Analyzer")

if "question" not in st.session_state:
    st.session_state.question = random.choice(java_questions)

st.markdown(f"### 🎯 Question:\n**{st.session_state.question}**")

uploaded_file = st.file_uploader("🎧 Upload your .mp3 answer", type=["mp3"])

transcribed_text = ""
if uploaded_file:
    with st.spinner("Transcribing..."):
        transcribed_text = transcribe_audio(uploaded_file)
    st.success("✅ Transcription complete.")
    st.markdown(f"**🗣 Your Answer:**\n{transcribed_text}")

if transcribed_text:
    query = st.session_state.question + " " + transcribed_text
    user_vector = embedder.encode(query).tolist()

    # Retrieve context from Pinecone
    search_result = index.query(vector=user_vector, top_k=3, include_metadata=True)
    chunks = [match.get("metadata", {}).get("text", "") for match in search_result["matches"]]
    context = "\n\n".join(chunks).strip()
    if not context:
        context = "No relevant documents were retrieved from the knowledge base."

    # Gemini evaluation
    prompt = f"""
You are an AI interview evaluator.

### QUESTION:
{st.session_state.question}

### CONTEXT:
{context}

### CANDIDATE'S ANSWER:
{transcribed_text}

Evaluate the answer on:
- Correctness
- Completeness
- Clarity
- Relevance

Give a score out of 5 and a short explanation.
"""
    with st.spinner("🔍 Evaluating your answer..."):
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        response = model.generate_content(prompt)

    st.markdown("## ✅ Evaluation Result")
    st.markdown(response.text)

    if st.button("🎲 New Question"):
        st.session_state.question = random.choice(java_questions)
        st.experimental_rerun()
