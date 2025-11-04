from flask import Flask
import os

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

@app.route("/")
def home():
    return "Flask is alive on Railway!"


# Lazy import (do this inside your API route)
@app.route("/analyze")
def analyze():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return "Model loaded!"
