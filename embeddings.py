import os
import numpy as np
import pickle
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Load the E5-Large embedding model
model = SentenceTransformer("intfloat/e5-large-v2")

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                texts.append(page.get_text())
            doc.close()
            print(f" Completed reading: {pdf_path}")
            print(f"number of pages in total are: {len(texts)}")
        except Exception as e:
            print(f" Error reading {pdf_path}: {e}")
    return texts

# Function to split text into chunks
def split_text_into_chunks(texts, max_chunk_size=1000):
    chunks = []
    for text in texts:
        words = text.split()
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) + 1 <= max_chunk_size:
                current_chunk += word + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    print(f"{len(chunks)} chunk created and continuing... \n")
                current_chunk = word + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
    print("chunks created")
    return chunks


# Function to create embeddings for a batch of texts
def create_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Embedding chunks: {i} to {i+batch_size}")
        try:
            batch_embeddings = model.encode(batch)  # Generate embeddings
            embeddings.extend(batch_embeddings)
            print(f" Processed batch {i + batch_size} / {len(texts)}")
        except Exception as e:
            print(f" Error creating embeddings for batch {i}: {e}")
            # Handle the error appropriately, e.g., skip the batch or exit
            # For example, to skip:
            # continue  # Skip to the next batch
            # Or to exit:
            # return None # Or raise the exception if you want to stop execution


    return np.array(embeddings) if embeddings else None # Return None if no embeddings were created


# Example PDF paths (replace with your actual paths)
pdf_paths = [
    "questions.pdf"
]

# Step 1: Extract text from PDFs
extracted_texts = extract_text_from_pdfs(pdf_paths)

# Step 2: Create chunks from the extracted texts
text_chunks = split_text_into_chunks(extracted_texts)

# Display the number of chunks and a few samples
print(f" Created {len(text_chunks)} text chunks.")
for i, chunk in enumerate(text_chunks[:min(4, len(text_chunks))]):  # Display first 4 or fewer
    print(f"Chunk {i + 1}: {chunk[:200]}...\n")  # Display first 200 chars

# Step 3: Generate embeddings for the text chunks
embeddings_array = create_embeddings(text_chunks)

if embeddings_array is not None: # Check if embeddings were successfully created
    # Step 4: Save text chunks to a file using Pickle
    with open('text_chunks.pkl', 'wb') as f:
        pickle.dump(text_chunks, f)
    print(" Text chunks saved successfully.")

    # Step 5: Save embeddings to NumPy file
    np.save('embeddings.npy', embeddings_array)
    print("Embeddings saved successfully as .npy")

    # Step 6: Save embeddings using Pickle
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings_array, f)
    print("Embeddings saved successfully as .pkl")

    # Step 7: Load embeddings later (Optional)
    loaded_embeddings = np.load('embeddings.npy')
    print(f"Loaded {loaded_embeddings.shape[0]} embeddings from NumPy.")

    with open('embeddings.pkl', 'rb') as f:
        loaded_embeddings_pickle = pickle.load(f)
    print(f"Loaded {len(loaded_embeddings_pickle)} embeddings from Pickle.")

    # Print sample embeddings (only if embeddings_array is not None)
    for i, record in enumerate(embeddings_array[:min(5, len(embeddings_array))]):  # Print first 5 or fewer
        print(f" Embedding {i + 1}: {record[:5]}...")  # Print first 5 dimensions
else:
    print("Embeddings generation failed.  Check the error messages above.")