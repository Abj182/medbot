from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from sentence_transformers import SentenceTransformer
import pdfplumber
from typing import List, Dict
import os
from pinecone import Pinecone, ServerlessSpec
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
from gtts import gTTS
import io
import base64
import requests  # For Perplexity API
from supabase import create_client, Client
from datetime import datetime
from functools import wraps
import bcrypt

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Change this!

# ============================================
# CONFIGURATION
# ============================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Set
PINECONE_INDEX_NAME = "medical-knowledge"
PINECONE_ENVIRONMENT = "us-east-1"

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in your environment variables
genai.configure(api_key=GEMINI_API_KEY)

# Perplexity API setup (for online search)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY") # Get from https://www.perplexity.ai/settings/api

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")  # Get from https://app.supabase.com
SUPABASE_KEY = os.getenv("SUPABASE_KEY") # Get from https://app.supabase.com

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Supabase connected")
except Exception as e:
    print(f"✗ Supabase connection error: {e}")
    supabase = None

# ============================================
# Embedding model
# ============================================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIMENSION = 384

casual_examples = [
    "hi", "hello", "hey", "how are you", "thanks", "thank you", "bye", "good morning", "good evening"
]
casual_embeddings = embedding_model.encode(casual_examples)

CASUAL_SIM_THRESHOLD = 0.7

# ============================================
# Pinecone KB Class
# ============================================
class PineconeMedicalKB:
    def __init__(self, api_key: str, index_name: str):
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
            print("✓ Index created")
        self.index = self.pc.Index(index_name)
        print(f"✓ Connected to Pinecone index: {index_name}")

    def get_stats(self):
        return self.index.describe_index_stats()

    def upload_from_pdf(self, pdf_path: str, chunk_size: int = 500):
        print(f"Processing PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text += f"[Page {i+1}] " + text + "\n"

        chunks = self._create_chunks(full_text, chunk_size)
        print(f"Created {len(chunks)} chunks")
        embeddings = embedding_model.encode(chunks, show_progress_bar=True)

        vectors_to_upsert = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors_to_upsert.append({
                "id": f"chunk_{i}",
                "values": emb.tolist(),
                "metadata": {"text": chunk[:1000], "chunk_index": i, "source": pdf_path}
            })
            if len(vectors_to_upsert) >= 100:
                self.index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
        print(f"✓ Uploaded {len(chunks)} chunks to Pinecone")
        return len(chunks)

    def _create_chunks(self, text: str, chunk_size: int) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size // 2):
            chunk = " ".join(words[i:i+chunk_size])
            if len(chunk) > 100:
                chunks.append(chunk)
        return chunks

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        embedding = embedding_model.encode([query])[0]
        results = self.index.query(vector=embedding.tolist(), top_k=top_k, include_metadata=True)
        formatted = []
        for match in results["matches"]:
            formatted.append({
                "text": match["metadata"]["text"],
                "similarity": match["score"],
                "chunk_index": match["metadata"].get("chunk_index", "N/A")
            })
        return formatted

    def clear_index(self):
        self.index.delete(delete_all=True)
        print("✓ Index cleared")

# ============================================
# Initialize Pinecone KB
# ============================================
print("=" * 60)
print("MEDICAL CHATBOT - INITIALIZING")
print("=" * 60)

try:
    kb = PineconeMedicalKB(PINECONE_API_KEY, PINECONE_INDEX_NAME)
    stats = kb.get_stats()
    total_vectors = stats.get("total_vector_count", 0)
    if total_vectors > 0:
        print(f"✓ Found {total_vectors} existing chunks in Pinecone")
    else:
        print("⚠️  No data in Pinecone yet. Upload a textbook via web interface.")
    print("✓ Knowledge base ready!")
except Exception as e:
    print(f"✗ Error connecting to Pinecone: {e}")
    kb = None

# Initialize Gemini model
try:
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("✓ Google Gemini initialized successfully")
except Exception as e:
    print(f"✗ Error initializing Gemini: {e}")
    print("Please check your GEMINI_API_KEY")
    gemini_model = None

print("=" * 60)

# ============================================
# Authentication Helper Functions
# ============================================
def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password: str, hashed: str) -> bool:
    """Check if password matches hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ============================================
# Chat History Helper Functions
# ============================================
def save_chat_message(user_id: int, chat_type: str, role: str, content: str):
    """Save a chat message to Supabase"""
    try:
        data = {
            "user_id": user_id,
            "chat_type": chat_type,  # 'textbook' or 'online'
            "role": role,  # 'user' or 'assistant'
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        supabase.table('chat_history').insert(data).execute()
    except Exception as e:
        print(f"Error saving chat: {e}")

def get_user_chat_history(user_id: int, chat_type: str = None, limit: int = 50):
    """Get user's chat history from Supabase"""
    try:
        query = supabase.table('chat_history').select('*').eq('user_id', user_id)
        
        if chat_type:
            query = query.eq('chat_type', chat_type)
        
        response = query.order('timestamp', desc=True).limit(limit).execute()
        return response.data
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return []

# ============================================
# Perplexity API Helper (Online Search)
# ============================================
def search_online_perplexity(query: str) -> Dict:
    """Search online using Perplexity API with focus on reliable medical sources"""
    if not PERPLEXITY_API_KEY or PERPLEXITY_API_KEY == "your-perplexity-api-key-here":
        return {
            "error": "Perplexity API key not configured",
            "answer": None,
            "citations": []
        }
    
    try:
        url = "https://api.perplexity.ai/chat/completions"
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a medical information assistant. Search for medical information from ONLY these reliable sources:
- World Health Organization (WHO)
- Centers for Disease Control and Prevention (CDC)
- National Health Service (NHS)
- Mayo Clinic
- PubMed/NIH
- MedlinePlus
- WebMD (medical reference sections only)

Provide accurate, evidence-based information with proper citations. Always include a disclaimer to consult healthcare professionals."""
                },
                {
                    "role": "user",
                    "content": f"Medical query: {query}\n\nPlease provide information from reliable medical sources only."
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_citations": True,
            "search_domain_filter": ["who.int", "cdc.gov", "nhs.uk", "mayoclinic.org", "nih.gov", "nlm.nih.gov", "webmd.com"],
            "return_images": False,
            "return_related_questions": False
        }
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        answer = data['choices'][0]['message']['content']
        citations = data.get('citations', [])
        
        return {
            "answer": answer,
            "citations": citations,
            "error": None
        }
        
    except Exception as e:
        print(f"Perplexity API Error: {e}")
        return {
            "error": str(e),
            "answer": None,
            "citations": []
        }

# ============================================
# gTTS Helper Function (FREE - No billing needed)
# ============================================
def generate_speech_gtts(text: str) -> bytes:
    """Generate speech using Google Translate TTS (completely free)"""
    try:
        # Clean text for speech
        clean_text = text.replace("⚠️", "Warning:").replace("📚", "").replace("🤖", "").strip()
        
        # Limit text length for TTS
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "..."
        
        # Generate speech using gTTS
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='com')
        
        # Save to bytes buffer
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        
        return audio_fp.read()
    
    except Exception as e:
        print(f"gTTS Error: {e}")
        return None

# ============================================
# Gemini Helper Function
# ============================================
def generate_response_with_gemini(user_question: str, context_chunks: list) -> str:
    """Generate medical response using Google Gemini"""
    if not gemini_model:
        return "⚠️ Gemini API not configured. Please add your API key."
    
    # Use top 3 chunks for context
    top_chunks = context_chunks[:3]
    context_text = "\n\n".join([
        f"Reference {i+1} (Relevance: {round(c['similarity']*100, 1)}%):\n{c['text']}" 
        for i, c in enumerate(top_chunks)
    ])

    prompt = f"""You are a knowledgeable medical assistant. Based on the medical textbook references provided below, answer the patient's question clearly and helpfully.

Medical Textbook References:
{context_text}

Patient's Question: {user_question}

Instructions:
1. Provide a clear, easy-to-understand summary based on the references
2. List possible conditions mentioned in the textbook
3. Suggest general care advice from the references
4. Mention when to seek immediate medical attention if relevant
5. Keep your response concise (under 250 words)
6. Only use information from the provided references
7. If the references don't adequately cover the question, say so clearly

Important: Always end by reminding the patient to consult a healthcare professional for proper diagnosis and treatment.

Your Response:"""

    try:
        # Generate response with Gemini
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500,
                top_p=0.8,
            )
        )
        
        return response.text
    
    except Exception as e:
        return f"⚠️ Error generating response with Gemini: {str(e)}\n\nPlease check your API key and internet connection."

# ============================================
# Flask Routes - Authentication
# ============================================
@app.route('/')
def landing():
    """Landing page - redirect to login or chat"""
    if 'user_id' in session:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'GET':
        return render_template('login.html')
    
    # Handle login POST
    data = request.json
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    try:
        # Check if user exists
        response = supabase.table('users').select('*').eq('email', email).execute()
        
        if not response.data:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        user = response.data[0]
        
        # Check password
        if not check_password(password, user['password_hash']):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Set session
        session['user_id'] = user['id']
        session['user_email'] = user['email']
        session['user_name'] = user['name']
        
        return jsonify({'success': True, 'redirect': '/chat'})
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'GET':
        return render_template('register.html')
    
    # Handle registration POST
    data = request.json
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    try:
        # Check if email already exists
        response = supabase.table('users').select('email').eq('email', email).execute()
        
        if response.data:
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        user_data = {
            'name': name,
            'email': email,
            'password_hash': hash_password(password),
            'created_at': datetime.utcnow().isoformat()
        }
        
        response = supabase.table('users').insert(user_data).execute()
        
        if response.data:
            user = response.data[0]
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            session['user_name'] = user['name']
            
            return jsonify({'success': True, 'redirect': '/chat'})
        
        return jsonify({'error': 'Registration failed'}), 500
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('login'))

# ============================================
# Flask Routes - Main App
# ============================================
@app.route('/chat')
@login_required
def chat():
    """Main chat interface"""
    return render_template("index.html", user_name=session.get('user_name', 'User'))

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if not kb or not gemini_model:
        return jsonify({"error": "Bot not fully initialized."}), 500

    data = request.json
    user_input = data.get("symptoms", "").strip()
    if not user_input:
        return jsonify({"error": "Please enter a message."}), 400

    user_id = session.get('user_id')
    
    # Save user message
    save_chat_message(user_id, 'textbook', 'user', user_input)

    # Check semantic similarity with casual messages
    user_emb = embedding_model.encode([user_input])[0]
    similarities = cosine_similarity([user_emb], casual_embeddings)[0]
    max_sim = max(similarities)

    if max_sim >= CASUAL_SIM_THRESHOLD:
        # Treat as friendly chat
        prompt = f"You are a friendly, helpful medical assistant. Respond warmly and briefly to this greeting or thank you message: '{user_input}'"
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=100,
                    top_p=0.9
                )
            )
            
            # Extract response text properly
            friendly_reply = None
            if hasattr(response, 'text') and response.text:
                friendly_reply = response.text
            
            # Fallback to predefined responses if Gemini fails
            if not friendly_reply:
                user_lower = user_input.lower()
                fallback_responses = {
                    "hi": "Hello! How can I assist you today?",
                    "hello": "Hi there! What can I help you with?",
                    "hey": "Hey! How's it going?",
                    "thank you": "You're welcome! Feel free to ask if you need anything else.",
                    "thanks": "No problem! Happy to help.",
                    "bye": "Goodbye! Take care and stay healthy!",
                    "good morning": "Good morning! How can I help you today?",
                    "good evening": "Good evening! What can I do for you?"
                }
                
                # Find closest match
                for key, value in fallback_responses.items():
                    if key in user_lower:
                        friendly_reply = value
                        break
                
                if not friendly_reply:
                    friendly_reply = "Hello! How can I help you today?"
            
            return jsonify({
                "query": user_input,
                "answer": friendly_reply,
                "references": [],
                "disclaimer": "",
                "llm_used": "Friendly Chat Mode",
                "audio": None
            })
            
        except Exception as e:
            print(f"Friendly chat error: {e}")
            
            # Fallback responses
            user_lower = user_input.lower()
            fallback_responses = {
                "hi": "Hello! How can I assist you today?",
                "hello": "Hi there! What can I help you with?",
                "hey": "Hey! How's it going?",
                "thank you": "You're welcome! Feel free to ask if you need anything else.",
                "thanks": "No problem! Happy to help.",
                "bye": "Goodbye! Take care and stay healthy!",
                "good morning": "Good morning! How can I help you today?",
                "good evening": "Good evening! What can I do for you?"
            }
            
            friendly_reply = "Hello! How can I help you today?"
            for key, value in fallback_responses.items():
                if key in user_lower:
                    friendly_reply = value
                    break
            
            return jsonify({
                "query": user_input,
                "answer": friendly_reply,
                "references": [],
                "disclaimer": "",
                "llm_used": "Friendly Chat Mode",
                "audio": None
            })

    # Treat as medical query
    total_vectors = kb.get_stats().get("total_vector_count", 0)
    if total_vectors == 0:
        return jsonify({
            "message": "No medical textbook uploaded yet. Please upload a textbook first.",
            "answer": None,
            "references": [],
            "disclaimer": "⚠️ Upload a medical textbook to get accurate information.",
            "audio": None
        })

    # Search Pinecone
    relevant_chunks = kb.search(user_input, top_k=5)
    filtered_chunks = [c for c in relevant_chunks if c["similarity"] > 0.3]

    if not filtered_chunks:
        return jsonify({
            "query": user_input,
            "answer": "I couldn't find relevant medical info. Please rephrase or consult a healthcare professional.",
            "references": [],
            "disclaimer": "⚠️ For educational purposes only.",
            "llm_used": "Google Gemini 2.0 Flash",
            "audio": None
        })

    # Generate medical response
    answer_text = generate_response_with_gemini(user_input, filtered_chunks)
    
    # Save assistant response
    save_chat_message(user_id, 'textbook', 'assistant', answer_text)
    
    # Generate audio for the response using gTTS (FREE)
    audio_base64 = None
    audio_bytes = generate_speech_gtts(answer_text)
    if audio_bytes:
        # Convert to base64 for sending to frontend
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    # Prepare references (top 3)
    references = [
        {
            "content": c["text"][:400] + "..." if len(c["text"]) > 400 else c["text"],
            "relevance": round(c["similarity"]*100, 2),
            "chunk_id": c.get("chunk_index", "N/A")
        }
        for c in filtered_chunks[:3]
    ]

    return jsonify({
        "query": user_input,
        "answer": answer_text,
        "references": references,
        "disclaimer": "⚠️ For educational purposes only. Always consult a healthcare professional.",
        "llm_used": "Google Gemini 2.0 Flash",
        "audio": audio_base64
    })

# ============================================
# Upload PDF Route
# ============================================
@app.route('/upload_textbook', methods=['POST'])
def upload_textbook():
    if not kb:
        return jsonify({"error": "Pinecone not available"}), 500
    
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    if file.filename.endswith(".pdf"):
        temp_path = "temp_medical_textbook.pdf"
        file.save(temp_path)
        try:
            chunks_created = kb.upload_from_pdf(temp_path)
            os.remove(temp_path)
            return jsonify({
                "message": "Medical textbook uploaded successfully to Pinecone!",
                "chunks_created": chunks_created,
                "storage": "Pinecone Cloud"
            })
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid file type. Please upload a PDF file."}), 400

# ============================================
# Stats Route
# ============================================
@app.route('/stats', methods=['GET'])
def get_stats():
    if not kb:
        return jsonify({"error": "Pinecone not connected"}), 500
    
    try:
        stats = kb.get_stats()
        return jsonify({
            "total_chunks": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", EMBEDDING_DIMENSION),
            "index_name": PINECONE_INDEX_NAME,
            "llm_provider": "Google Gemini 2.0 Flash + gTTS"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================
# Clear Index Route
# ============================================
@app.route('/clear', methods=['POST'])
def clear_index():
    if not kb:
        return jsonify({"error": "Pinecone not connected"}), 500
    
    try:
        kb.clear_index()
        return jsonify({"message": "Pinecone index cleared successfully. All textbook data removed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================
# Online Search Route
# ============================================
@app.route('/search_online', methods=['POST'])
@login_required
def search_online():
    """Search online medical information using Perplexity API"""
    data = request.json
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Please provide a search query"}), 400
    
    user_id = session.get('user_id')
    
    # Save user query
    save_chat_message(user_id, 'online', 'user', query)
    
    try:
        # Search using Perplexity
        result = search_online_perplexity(query)
        
        if result["error"]:
            error_msg = f"⚠️ Online search error: {result['error']}"
            save_chat_message(user_id, 'online', 'assistant', error_msg)
            return jsonify({
                "query": query,
                "answer": error_msg,
                "citations": [],
                "disclaimer": "Unable to search online. Please try again or consult a healthcare professional.",
                "source": "Online Search (Error)"
            })
        
        # Save assistant response
        save_chat_message(user_id, 'online', 'assistant', result["answer"])
        
        # Generate audio for the response
        audio_base64 = None
        if result["answer"]:
            audio_bytes = generate_speech_gtts(result["answer"])
            if audio_bytes:
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return jsonify({
            "query": query,
            "answer": result["answer"],
            "citations": result["citations"],
            "disclaimer": "⚠️ Information from online sources. Always verify with a healthcare professional.",
            "source": "Online Search (Perplexity)",
            "audio": audio_base64
        })
        
    except Exception as e:
        print(f"Online search error: {e}")
        return jsonify({
            "error": f"Search failed: {str(e)}"
        }), 500

# ============================================
# Chat History Routes
# ============================================
@app.route('/api/chat_history', methods=['GET'])
@login_required
def get_chat_history():
    """Get user's chat history"""
    user_id = session.get('user_id')
    chat_type = request.args.get('type')  # 'textbook' or 'online' or None for all
    limit = request.args.get('limit', 50, type=int)
    
    history = get_user_chat_history(user_id, chat_type, limit)
    
    return jsonify({
        'success': True,
        'history': history
    })

@app.route('/api/clear_history', methods=['POST'])
@login_required
def clear_history():
    """Clear user's chat history"""
    user_id = session.get('user_id')
    chat_type = request.json.get('type')  # 'textbook', 'online', or 'all'
    
    try:
        if chat_type == 'all':
            supabase.table('chat_history').delete().eq('user_id', user_id).execute()
        else:
            supabase.table('chat_history').delete().eq('user_id', user_id).eq('chat_type', chat_type).execute()
        
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        print(f"Error clearing history: {e}")
        return jsonify({'error': 'Failed to clear history'}), 500

# ============================================
# Online Search Route
# ============================================

@app.route('/search_online', methods=['POST'])
def search_online():
    """Search online medical information using Perplexity API"""
    data = request.json
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Please provide a search query"}), 400
    
    try:
        # Search using Perplexity
        result = search_online_perplexity(query)
        
        if result["error"]:
            return jsonify({
                "query": query,
                "answer": f"⚠️ Online search error: {result['error']}",
                "citations": [],
                "disclaimer": "Unable to search online. Please try again or consult a healthcare professional.",
                "source": "Online Search (Error)"
            })
        
        # Generate audio for the response
        audio_base64 = None
        if result["answer"]:
            audio_bytes = generate_speech_gtts(result["answer"])
            if audio_bytes:
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return jsonify({
            "query": query,
            "answer": result["answer"],
            "citations": result["citations"],
            "disclaimer": "⚠️ Information from online sources. Always verify with a healthcare professional.",
            "source": "Online Search (Perplexity)",
            "audio": audio_base64
        })
        
    except Exception as e:
        print(f"Online search error: {e}")
        return jsonify({
            "error": f"Search failed: {str(e)}"
        }), 500
# ============================================
# Run App (keep at end of file)
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

