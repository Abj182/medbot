from flask import Flask
import os

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

@app.route("/")
def home():
    return "✅ Flask is working on Railway!"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
