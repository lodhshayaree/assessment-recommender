import os
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Load model and data
embedder = SentenceTransformer("all-MiniLM-L6-v2")
catalog_df = pd.read_csv("model/catalog.csv")
catalog_embeddings = embedder.encode(catalog_df["assessment_description"].tolist(), show_progress_bar=True)

# Flask app setup
app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

def refine_query_with_gemini(query):
    prompt = f"Rewrite this user query into a formal, structured format suitable for assessment matching: '{query}'"
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def get_top_recommendations(user_query, top_k=5):
    refined_query = refine_query_with_gemini(user_query)
    query_embedding = embedder.encode([refined_query])[0]
    similarities = cosine_similarity([query_embedding], catalog_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = catalog_df.iloc[top_indices]
    scores = similarities[top_indices]
    results["similarity_score"] = scores
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        results = get_top_recommendations(query)
        return render_template("results.html", query=query, results=results)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

