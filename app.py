import os
import requests
import json
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
  """Example Hello World route."""
  name = os.environ.get("NAME", "World")
  return f"Hello {name}!"


#@app.route('/reply', methods=['POST'])
@app.route('/reply')
def reply():
  #data = request.get_json()
  query = "Tell me about Fintech Olympiad"
  sentiment = analyze_sentiment(query)
  if(sentiment == "True"):
    print("Angry")
  else:
    print("Not Angry")
  #query = data.get('query')
  #name = data.get('name')
  #unique_id = data.get('unique_id')
  faq = get_docs()
  response = get_response(faq, query)
  return jsonify({"reply": response})
  
def analyze_sentiment(query):
    messages = [
        SystemMessage(content="You are a helpful AI Agent who is an expert at analyzing sentiment. Return 'True' if the user is angry, annoyed or frustrated. Else return 'False'. You don't give explanaitions. You answer in one word: 'True' or 'False'."),
        HumanMessage(content="What is the sentiment of: " + query),
    ]

    #print("\n--- Relevant Documents ---")
    #for i, doc in enumerate(relevant_docs, 1):
    #    print(f"Document {i}:\n{doc.page_content}\n")

    model = ChatGroq(model="llama3-70b-8192", api_key = "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")

    sentiment = model.invoke(messages)
    return sentiment.content

def get_docs():
    url = "https://script.google.com/macros/s/AKfycbzfXyszyHs4YHIxGXocaXqjzpNjZsOOIzxdGVfQ0ZHv_9M-hFAJvVGF8pIkBcVnUldVlQ/exec"

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
    return faq

    
def get_response(faq, query):
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n" + faq
        + "\n\nPlease provide an answer based only on the provided documents. The answer should be crisp and concise but should contain all of the information."
        + "\n\nYour response should not be rude. Don't let the user know which document you are referring to. While answering, don't make things on your own, stick to the facts in the document."
        + "\n\nIf you don't know the answer then refrain from answering anything!"
    )

    messages = [
        SystemMessage(content="You are a helpful AI Agent. You don't anything if you don't know the answer. You don't give any clarification. You answer with None. If you are answering with None, you just give a one word answer 'None'."),
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
