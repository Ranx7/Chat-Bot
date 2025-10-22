from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)
CORS(app)  # ✅ allows cross-domain requests (Vercel → Render)

# Load your Q&A data
with open("qa_data.json", "r") as f:
    qa_data = json.load(f)

questions = [item["question"] for item in qa_data]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarity.argmax()
    best_score = similarity[0, best_match_index]

    if best_score < 0.2:
        return "I'm not sure about that. Could you rephrase your question?"
    else:
        return qa_data[best_match_index]["answer"]

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("message", "")
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
