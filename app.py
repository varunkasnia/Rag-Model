from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import torch
import os

app = Flask(__name__)
SAVE_PATH = "rag_files"

# Global variables for the loaded RAG components
generator_tokenizer = None
generator_model = None
retriever_model = None
faiss_index = None
knowledge_data = None

# --- Model Loading Function ---
def load_rag_components():
    global generator_tokenizer, generator_model, retriever_model, faiss_index, knowledge_data

    # 1. Load Generator Model and Tokenizer
    try:
        # Load from the local directory
        generator_tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
        generator_model = AutoModelForCausalLM.from_pretrained(SAVE_PATH)
    except Exception as e:
        print(f"Error loading generator model: {e}")
        # Fallback/Debug: Load a base model if local fails
        # generator_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        # generator_model = AutoModelForCausalLM.from_pretrained('distilgpt2')

    # 2. Load Retriever (Embedding Model)
    # The SentenceTransformer model itself is usually loaded from the hub/cache
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2' # MUST match the one used for saving!
    retriever_model = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Load FAISS Index
    faiss_index = faiss.read_index(f"{SAVE_PATH}/faiss_index.bin")

    # 4. Load Knowledge Base Data
    # Load the specific dataset split you saved
    knowledge_data = load_from_disk(f"{SAVE_PATH}/knowledge_base")['knowledge']
    
    # Set the model to evaluation mode
    generator_model.eval()

# --- RAG Inference Function (Adapted from previous steps) ---
def rag_inference(query, top_k=2):
    # 1. Retrieve Context
    query_vector = retriever_model.encode(query)
    # Search the FAISS index
    D, I = faiss_index.search(np.expand_dims(query_vector, axis=0), top_k)
    
    retrieved_texts = [knowledge_data[i.item()]['text'] for i in I[0]]
    context = "\n---\n".join(retrieved_texts)

    # 2. Generate Answer
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    input_ids = generator_tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad(): # Disable gradient calculation for faster inference
        output_ids = generator_model.generate(
            input_ids,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=generator_tokenizer.eos_token_id
        )

    response = generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Clean up the output
    try:
        answer_start = response.index("Answer:") + len("Answer:")
        final_answer = response[answer_start:].strip()
    except ValueError:
        final_answer = response.strip()
        
    return final_answer, retrieved_texts

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    context_chunks = None
    query = None
    
    if request.method == 'POST':
        query = request.form['query']
        if query:
            try:
                answer, context_chunks = rag_inference(query)
            except Exception as e:
                answer = f"An error occurred during inference: {e}"
                context_chunks = ["N/A"]

    return render_template(
        'index.html', 
        answer=answer, 
        context_chunks=context_chunks, 
        query=query
    )

if __name__ == '__main__':
    # Load components once when the server starts
    load_rag_components()
    app.run(debug=True)