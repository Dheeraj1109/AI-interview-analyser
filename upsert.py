import os
import numpy as np
import pickle
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
pine_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(pine_api_key, environment='us-east-1')

# Define the index name
index_name = "java"

# Load previously saved text chunks
text_chunks_file = 'text_chunks.pkl'
embeddings_file = 'embeddings.npy'

if not os.path.exists(text_chunks_file) or not os.path.exists(embeddings_file):
    print(" Error: One or both files (text_chunks.pkl, embeddings.npy) are missing.")
    exit(1)

# Load text chunks
with open(text_chunks_file, 'rb') as f:
    text_chunks = pickle.load(f)

# Load embeddings
embeddings = np.load(embeddings_file)

# Ensure the number of text chunks matches embeddings
if len(text_chunks) != len(embeddings):
    print(f" Mismatch detected: {len(text_chunks)} text chunks vs {len(embeddings)} embeddings.")
    exit(1)

print(f" Loaded {len(text_chunks)} text chunks and {len(embeddings)} embeddings.")

# Create the index if it doesn't exist
try:
    existing_indexes = pc.list_indexes()

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=embeddings.shape[1],  # Auto-detect embedding size
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f" Index '{index_name}' created successfully.")
    else:
        print(f" Index '{index_name}' already exists.")
except Exception as e:
    print(" Index creation failed:", e)
    exit(1)

# Wait for the index to be ready
while True:
    index_status = pc.describe_index(index_name)
    if index_status.status.state == "Ready":
        break
    time.sleep(1)

print(" Pinecone Serverless Index is Ready!")

# Connect to the index
index = pc.Index(index_name)

# Prepare data for upsert
batch_size = 100  # Adjust batch size as needed
data = [
    {
        'id': str(i),
        'values': embeddings[i].tolist(),
        'metadata': {'text': text_chunks[i]}
    }
    for i in range(len(embeddings))
]

# Upload data in batches
for i in range(0, len(data), batch_size):
    batch_data = data[i:i + batch_size]

    try:
        index.upsert(
            vectors=batch_data,
            namespace="example-namespace"  # Specify your namespace
        )
        print(f" Upserted batch {i // batch_size + 1} successfully.")
    except Exception as e:
        print(f" Error upserting batch {i // batch_size + 1}: {e}")

    time.sleep(1)  # Optional: wait between batches

# Print index stats
print(" Index Stats:", index.describe_index_stats())