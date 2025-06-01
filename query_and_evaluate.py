import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone  # ✅ correct import
import random as rd
import speechtotext as stt

# === 1. Configure your APIs ===
genai.configure(api_key="AIzaSyAc_cTPTtI7QmG08w99fKKingFKuHRl9q8")

# ✅ Initialize Pinecone
pc = Pinecone(api_key="pcsk_4twyfY_DwDPWiJL826wcfZREc6wbd8cNpTkw6ocLTML4g19gMxd57yyLgNGPD18WDFkGQ1")
index = pc.Index("java")



# === 2. Get user input ===
javaques = ['What is Java', 'What is JVM', 'What is JRE', 'What is JDK', 'What is bytecode', 'Is Java a compiled or interpreted language', 'What is a class in Java', 'What is an object in Java', 'What is the main method in Java', 'What are data types in Java', 'What are Java access modifiers', 'What is a constructor', 'What is method overloading', 'What is method overriding', 'What is inheritance in Java', 'What is polymorphism', 'What is abstraction', 'What is encapsulation', 'What is the difference between == and equals()', 'What are wrapper classes in Java', 'What is an interface', 'What is an abstract class', 'What is the difference between abstract class and interface', 'What is the use of final keyword', 'What is the difference between static and non-static methods', 'What is the this keyword', 'What is the super keyword', 'What are packages in Java', 'What is exception handling', 'What are checked and unchecked exceptions', 'What is a try-catch block', 'What is a finally block', 'What is throw and throws', 'What is garbage collection', 'What is the difference between List and Set', 'What is HashMap in Java', 'What is the difference between HashMap and Hashtable', 'What is synchronization', 'What is multithreading', 'What is the difference between process and thread', 'What is the Runnable interface', 'How to create a thread in Java', 'What is the difference between wait() and sleep()', 'What is deadlock', 'What is a lambda expression', 'What is functional interface', 'What are streams in Java 8', 'What is Optional in Java 8', 'What is method reference', 'What is serialization', 'What is the transient keyword', 'What is reflection in Java', 'What is a singleton class', 'How to make a class immutable', 'What is the difference between equals() and hashCode()', 'What is the Java Memory Model', 'What is a volatile variable', 'What is the default value of int', 'What is autoboxing', 'What is unboxing', 'What is the diamond operator', 'What is an enum', 'What is a marker interface', 'What is the difference between fail-fast and fail-safe', 'What is the difference between ArrayList and Vector', 'What is the difference between Comparator and Comparable', 'What is cloning in Java', 'What is shallow and deep copy', 'What is the use of instanceof', 'What is method hiding', 'Can we override static methods', 'Can we override private methods', 'What is composition', 'What is aggregation', 'What is association', 'What is dependency injection', 'What is the use of the new keyword', 'What is a memory leak in Java', 'What is the finalize() method', 'What is method chaining', 'What is a constructor overloading', 'What are varargs', 'What is recursion', 'What is a nested class', 'What is a static nested class', 'What is annotation', 'What is a custom annotation', 'What is JavaDoc', 'What is the default package in Java', 'What is the difference between path and classpath', 'What is the difference between public, private, protected', 'What are static imports', 'What is the main difference between Stack and Queue', 'What is the difference between throw and throws', 'What is File class in Java', 'What is JavaBeans', 'What is the difference between error and exception', 'What is the purpose of the instanceof operator']
#use random to as any random question from JAVA

embedder = SentenceTransformer("all-MiniLM-L6-v2")

for i in range(3):
    question = rd.choice(javaques)
    print(f"\n🧠 Java Interview Question {i+1}:\n{question}")

    use_audio_file = input("\nDo you want to upload an MP3 file? (y/n): ").strip().lower()

    if use_audio_file == 'y':
        audio_path = input("Enter path to your .mp3 file: ").strip()
        user_input = stt.stt(audio_path=audio_path)
    else:
        print("Recognizing speech from microphone...")
        user_input = stt.stt()

    print(f"\n🗣 Transcribed Answer: {user_input}")
    confirm = input("Press Enter to continue, or type a corrected version: ").strip()
    if confirm:
        user_input = confirm

    query = question + " " + user_input
    user_vector = embedder.encode(query).tolist()

    search_result = index.query(vector=user_vector, top_k=3, include_metadata=True)

    chunks = []
    print("\n📚 Retrieved Context from Pinecone:")
    if not search_result["matches"]:
        print("❌ No matching documents found.")
    else:
        for j, match in enumerate(search_result["matches"], 1):
            text = match.get("metadata", {}).get("text", "")
            score = match.get("score", 0)
            print(f"\n🔹 Chunk {j} (Score: {score:.4f}):\n{text}")
            if text:
                chunks.append(text)

    context = "\n\n".join(chunks).strip()
    if not context:
        context = "No relevant documents were retrieved from the knowledge base."

    prompt = f"""
    You are an AI interview evaluator.

    Below is a candidate's spoken answer to a Java interview question, and the most relevant documents retrieved using RAG (retrieval augmented generation).

    ### QUESTION:
    {question}

    ### CONTEXT (retrieved from the knowledge base):
    {context}

    ### CANDIDATE'S ANSWER (transcribed):
    {user_input}

    Evaluate the answer on:
    - Correctness
    - Completeness
    - Clarity
    - Relevance

    Give a score out of 5 and a short explanation.
    """

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content(prompt)

    print(f"\n✅ --- AI Evaluation for Question {i+1} ---")
    print(response.text)

    # Log to file
    with open("evaluation_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Q{i+1}: {question}\n")
        f.write(f"A: {user_input}\n")
        f.write(f"Context: {context}\n")
        f.write(f"Evaluation:\n{response.text}\n")
        f.write("-" * 60 + "\n\n")
