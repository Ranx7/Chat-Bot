from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests

app = Flask(__name__)
CORS(app)

# API CALL COUNTERS
stats = {
    "total_calls": 0,
    "corpus_calls": 0,
    "web_calls": 0
}

# Load Local Corpus (Mode 1)
with open("qa_data.json", "r") as f:
    qa_data = json.load(f)

questions = [item["question"] for item in qa_data]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


# ---------- FIXED CORPUS ANSWER ----------
def corpus_answer(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)

    best_idx = similarity.argmax()
    best_score = similarity[0, best_idx]

    # Return structured response
    if best_score < 0.25:  # LOWERED to reduce false matches
        return {
            "answer": None,
            "low_confidence": True
        }

    return {
        "answer": qa_data[best_idx]["answer"],
        "low_confidence": False
    }


# ---------- Web Search (Mode 2) ----------
SERPER_KEY = "YOUR_API_KEY_HERE"

def web_search(query):
    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_KEY,
        "Content-Type": "application/json",
    }

    payload = json.dumps({"q": query})
    response = requests.post(url, headers=headers, data=payload)
    data = response.json()

    # Priority 1 — AnswerBox
    if "answerBox" in data:
        ans = (
            data["answerBox"].get("answer") or
            data["answerBox"].get("snippet")
        )
        if ans:
            return ans

    # Priority 2 — Knowledge Graph
    if "knowledgeGraph" in data:
        desc = data["knowledgeGraph"].get("description")
        if desc:
            return desc

    # Priority 3 — Organic Search
    if "organic" in data and len(data["organic"]) > 0:
        return data["organic"][0].get("snippet", "No snippet available")

    return "No answer found online."


# ---------- MAIN ENDPOINT /ask ----------
@app.route("/ask", methods=["POST"])
def ask():
    global stats

    data = request.get_json()

    user_input = data.get("message", "")
    mode = data.get("mode", "corpus")
    include_stats = data.get("include_stats", False)

    stats["total_calls"] += 1

    # MODE = WEB → ALWAYS web search
    if mode == "web":
        stats["web_calls"] += 1
        response = web_search(user_input)

    # MODE = CORPUS → Check confidence
    elif mode == "corpus":
        stats["corpus_calls"] += 1
        result = corpus_answer(user_input)

        if result["low_confidence"]:
            response = "I'm not sure. Try web search mode for better results."
        else:
            response = result["answer"]

    else:
        response = "Invalid mode. Use 'corpus' or 'web'."

    payload = {
        "mode_used": mode,
        "response": response
    }

    if include_stats:
        payload["stats"] = stats

    return jsonify(payload)


# ---------- OPTIONAL: GET /stats ----------
@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify(stats)


if __name__ == "__main__":
    app.run(debug=True)

# made by yours truly ranx <3
