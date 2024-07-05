import os
import requests
import json
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
  """Example Hello World route."""
  name = os.environ.get("NAME", "World")
  return f"Hello {name}!"

@app.route('/reply', methods=['POST'])
#@app.route('/reply')
def reply():
  #data = request.get_json()
  query = "Who was the founder of Microsoft?"
  #query = data.get('query')
  #name = data.get('name')
  #unique_id = data.get('unique_id')
  faq, context, answer = get_docs()
  db = create_vector_store(faq)
  relevant_docs = get_relevant_doc(db, query)
  response = get_response(relevant_docs, query)
  return jsonify({"reply": response})

def get_docs():
    url = "https://script.google.com/macros/s/AKfycbyGC3gvjFJM-GHO6RysDr4sjtiGpQkOj--aFLyq6hS8Qckvf6SYecU4jicLW0xZlCewWg/exec"

    payload = {}
    headers = {
    'Cookie': 'NID=514=bYRBziWOaU8u6ymcQhZt9H_nnm3rVkAu6bbQv03hDn2mhRky1A5NR8GbsGJp7eMrN_AR3MEz0RpOPrvJK8asZmOTmyF-87acCVt2VOI3gXhMlw2drMdW0OCoz3hoIUSUnfcn8GuQjAN1bQiQkdp4WjZ5FIfgWoT0lG0nYcI5N-M'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    response_text = response.text
    data = json.loads(response_text)
    faq = data.get("FAQ")
    context = data.get("Context")
    answer = data.get("Answer")
    return faq, context, answer

def create_vector_store(faq):
    model = ChatGroq(model="llama3-70b-8192", api_key = "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")


    # Define the directory containing the text file and the persistent directory
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "Microsoft.txt")
    #file_path = os.path.join(current_dir, "fintech", "Microsoft1000.txt")
    persistent_directory = os.path.join(current_dir, "chroma_db")

    # Read the text content from the file
    loader = TextLoader(file_path)
    #documents = loader.load()
    documents = faq
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(documents)
    # Display information about the split documents
    #print("\n--- Document Chunks Information ---")
    #print(f"Number of document chunks: {len(docs)}")
    #print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    #print("\n--- Creating embeddings ---")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    #print("\n--- Creating vector store ---")
    db = Chroma.from_texts(docs, embedding_function)
    #print("\n--- Finished creating vector store ---")
    return db

def get_relevant_doc(db, query):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the existing vector store with the embedding function
    #db = Chroma(persist_directory="./chroma_db",
    #            embedding_function=embedding_function)

    #query = "Who founded Microsoft?"

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    relevant_docs = retriever.invoke(query)
    return relevant_docs
    
def get_response(relevant_docs, query):
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide an answer based only on the provided documents."
        + "\n\nYour response should not be rude. Don't let the user know which document you are referring to."
        + "\n\nIf you don't know the answer then refrain from answering anything!"
    )

    messages = [
        SystemMessage(content="Rounds"),
        HumanMessage(content=combined_input),
    ]

    #print("\n--- Relevant Documents ---")
    #for i, doc in enumerate(relevant_docs, 1):
    #    print(f"Document {i}:\n{doc.page_content}\n")

    model = ChatGroq(model="llama3-70b-8192", api_key = "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")

    result = model.invoke(messages)
    return result.content

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
