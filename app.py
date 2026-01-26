import os
import qrcode
import base64
import json
import csv
import io
import uuid
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from scoring import score_product_from_db
from custom_scoring import calculate_custom_scores_from_db
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import blockchain  # Blockchain Score Registry
from seed_data import SEED_PRODUCTS, SEED_USERS, generate_search_links  # Seed data for database

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ==========================================
# DATABASE CONFIGURATION (Hybrid: SQLite local / PostgreSQL production)
# ==========================================
# Use PostgreSQL if DATABASE_URL exists (Render production), else SQLite (local dev)
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    # Render provides postgres:// but SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
else:
    # Local development - use SQLite
    DB_PATH = os.path.join(BASE_DIR, 'db.sqlite3')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret')

# ==========================================
# SESSION TIMEOUT CONFIGURATION
# ==========================================
# Default session timeout: 30 minutes of inactivity
# Remember Me: 7 days
SESSION_TIMEOUT_MINUTES = 30
REMEMBER_ME_DAYS = 7

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=REMEMBER_ME_DAYS)

# Database
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'error'


# ==========================================
# SESSION TIMEOUT HANDLING
# ==========================================
@app.before_request
def check_session_timeout():
    """Check for session inactivity timeout."""
    # Skip for static files and non-authenticated users
    if request.endpoint == 'static' or not current_user.is_authenticated:
        return
    
    # Skip for login/logout/signup routes
    if request.endpoint in ['login', 'logout', 'signup', 'home']:
        return
    
    # Check if user chose "Remember Me"
    if session.get('remember_me'):
        # For "Remember Me" users, just update last activity (no timeout)
        session['last_activity'] = datetime.now().isoformat()
        return
    
    # Check inactivity timeout for non-remembered sessions
    last_activity = session.get('last_activity')
    
    if last_activity:
        last_activity_time = datetime.fromisoformat(last_activity)
        inactive_duration = datetime.now() - last_activity_time
        
        if inactive_duration > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            # Session expired due to inactivity
            logout_user()
            session.clear()
            flash(f'Your session expired after {SESSION_TIMEOUT_MINUTES} minutes of inactivity. Please log in again.', 'error')
            return redirect(url_for('login'))
    
    # Update last activity time
    session['last_activity'] = datetime.now().isoformat()


# ==========================================
# GOOGLE GEMINI AI PHONE SCANNER
# ==========================================

def identify_phone_with_gemini(image_base64):
    """
    Use Google Gemini API to identify a phone from an image.
    Returns the identified phone model name.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not configured", "phone": None}
    
    # Try multiple model endpoints (in case one doesn't work)
    models_to_try = [
        "gemini-2.0-flash",      # Current stable
        "gemini-2.5-flash",      # Latest
        "gemini-pro-latest",     # Pro model
    ]
    
    # Prepare the prompt
    prompt = """Analyze this image and identify the smartphone shown.

IMPORTANT: Respond ONLY with a JSON object in this exact format:
{
    "identified": true/false,
    "brand": "Apple" or "Samsung" or other brand name,
    "model": "exact model name like iPhone 15 Pro Max or Galaxy S24 Ultra",
    "confidence": "high", "medium", or "low",
    "details": "brief description"
}

If no phone is visible, set "identified" to false."""

    # Prepare request body
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_base64
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "topK": 1,
            "topP": 1,
            "maxOutputTokens": 1024,
        }
    }
    
    last_error = None
    
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        
        try:
            response = requests.post(url, json=body, timeout=30)
            
            # If 404 or 429, try next model
            if response.status_code == 404:
                last_error = f"Model {model_name} not available"
                continue
            
            if response.status_code == 429:
                last_error = f"Rate limited - please wait a moment and try again"
                continue
                
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the text response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                
                # Check for safety blocks
                if "content" not in candidate:
                    finish_reason = candidate.get("finishReason", "unknown")
                    if finish_reason == "SAFETY":
                        last_error = "Image was blocked by safety filters. Try a clearer photo."
                        continue
                    last_error = f"No content returned: {finish_reason}"
                    continue
                
                text = candidate["content"]["parts"][0]["text"]
                
                # Parse JSON from response (handle markdown code blocks)
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                
                # Try to parse JSON
                import re
                try:
                    parsed = json.loads(text)
                    return parsed
                except json.JSONDecodeError as je:
                    # Try to extract brand and model from truncated JSON
                    brand_match = re.search(r'"brand"\s*:\s*"([^"]+)"', text)
                    model_match = re.search(r'"model"\s*:\s*"([^"]*)', text)
                    
                    if brand_match:
                        brand = brand_match.group(1)
                        model = model_match.group(1) if model_match else "Unknown"
                        return {
                            "identified": True,
                            "brand": brand,
                            "model": model,
                            "details": "Identified from image"
                        }
                    
                    # If all else fails
                    return {
                        "identified": False,
                        "error": "AI response format issue",
                        "details": text[:200] if text else "No response text"
                    }
            
            return {"error": "No response from AI", "phone": None}
            
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            continue
        except Exception as e:
            last_error = str(e)
            continue
    
    # All models failed
    return {"error": f"All Gemini models failed. Last error: {last_error}. Please check your API key at https://aistudio.google.com/app/apikey", "phone": None}


def match_phone_in_database(brand, model):
    """
    Match the identified phone with products in the database.
    Uses strict matching to ensure correct phone is returned.
    Returns None if no confident match found.
    """
    if not brand or not model:
        return None
    
    # Normalize the search terms
    brand_lower = brand.lower().strip()
    model_lower = model.lower().strip()
    
    # Extract key identifiers from model name
    import re
    
    # Get all products
    products = Product.query.all()
    
    best_match = None
    best_score = 0
    brand_matched = False
    
    # Extract model series/number (e.g., "M32", "S24", "15 Pro", "A55")
    model_identifier = re.findall(r'[a-zA-Z]?\d+(?:\s*(?:pro|max|ultra|plus|mini|lite|fe|s|e))*', model_lower, re.IGNORECASE)
    model_identifier_str = ' '.join(model_identifier).strip()
    
    for product in products:
        score = 0
        product_name_lower = product.product_name.lower()
        company_lower = (product.company_name or "").lower()
        
        # Check brand match FIRST - if brand doesn't match, skip this product
        brand_matches = (brand_lower in company_lower or company_lower in brand_lower or
                        brand_lower in product_name_lower)
        
        if not brand_matches:
            continue  # Different brand, skip entirely
        
        brand_matched = True
        score += 30  # Base score for brand match
        
        # Extract model numbers from both
        model_numbers = re.findall(r'\d+', model_lower)
        product_numbers = re.findall(r'\d+', product_name_lower)
        
        # Check for exact model number match (e.g., "32" in M32, "24" in S24)
        number_matches = 0
        for num in model_numbers:
            if num in product_numbers:
                number_matches += 1
                score += 40  # Strong signal for number match
        
        # Check for model series letter match (e.g., "M" in M32, "S" in S24, "A" in A55)
        series_match = re.search(r'([a-zA-Z])\s*\d+', model_lower)
        if series_match:
            series_letter = series_match.group(1).lower()
            # Check if same series letter appears in product name before a number
            product_series = re.search(r'([a-zA-Z])\s*\d+', product_name_lower)
            if product_series:
                if series_letter == product_series.group(1).lower():
                    score += 50  # Series match (e.g., both are "M" series or "S" series)
                else:
                    score -= 30  # Different series (M32 vs S24) - strong penalty
        
        # Check for "Pro", "Max", "Ultra", "Plus", "Mini", "Lite", "FE" variants
        variants = ["pro", "max", "ultra", "plus", "mini", "lite", "fe", "neo"]
        for variant in variants:
            model_has = variant in model_lower
            product_has = variant in product_name_lower
            if model_has and product_has:
                score += 30  # Both have the variant
            elif model_has and not product_has:
                score -= 40  # Model has variant but product doesn't - wrong variant
            elif not model_has and product_has:
                score -= 40  # Product has variant but model doesn't - wrong variant
        
        # Exact product name contains model name or vice versa
        if model_lower in product_name_lower:
            score += 80  # Strong match
        
        # Check for special keywords like "Galaxy", "iPhone", "Pixel", "Redmi", etc.
        keywords = ["galaxy", "iphone", "pixel", "redmi", "note", "poco", "realme", "oneplus", "motorola", "moto"]
        for keyword in keywords:
            if keyword in model_lower and keyword in product_name_lower:
                score += 20
            elif keyword in model_lower and keyword not in product_name_lower:
                score -= 20  # Keyword mismatch
        
        if score > best_score:
            best_score = score
            best_match = product
    
    # STRICT THRESHOLD: Require high confidence match
    # Brand match alone (30) is NOT enough
    # Need brand (30) + at least model number match (40) + series match (50) = 120 for confident match
    # Minimum threshold of 100 ensures we have more than just brand match
    if best_score >= 100:
        return best_match
    
    return None


# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'consumer' or 'producer'
    company_name = db.Column(db.String(128), nullable=True)  # For producers: 'Apple', 'Samsung', etc.
    
    def __repr__(self):
        return f'<User {self.username} ({self.role})>'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Hugging Face Emotion Detection Client
hf_client = None
try:
    hf_client = InferenceClient(
        provider="hf-inference",
        api_key=os.environ.get("HF_TOKEN"),
    )
except Exception as e:
    pass  # HF client initialization failed, will retry on first use

# Emotion to score mapping (1-5 scale)
# Based on michellejieli/emotion_text_classifier model outputs
# These represent the sentiment polarity of each emotion
EMOTION_SCORES = {
    'joy': 5.0,           # Very positive
    'surprise': 3.5,      # Can be positive or negative, slightly positive bias
    'neutral': 3.0,       # Neutral
    'sadness': 2.0,       # Negative
    'fear': 1.5,          # Very negative
    'disgust': 1.0,       # Very negative
    'anger': 1.0,         # Very negative
}

# Emotion categories for credibility analysis
POSITIVE_EMOTIONS = {'joy'}
NEUTRAL_EMOTIONS = {'neutral', 'surprise'}
NEGATIVE_EMOTIONS = {'sadness', 'fear', 'disgust', 'anger'}

def analyze_emotion(text):
    """
    Analyze emotion from text and return emotion details.
    Returns: dict with 'score', 'label', 'confidence', and 'credibility_flag'
    """
    default_result = {
        'score': 3.0,
        'label': 'neutral',
        'confidence': 0.0,
        'all_emotions': []
    }
    
    if not text:
        return default_result
    
    # Ensure HF client is initialized and has a token
    global hf_client
    token = os.environ.get("HF_TOKEN")
    if (not hf_client) or (token and getattr(hf_client, "_api_key", None) in (None, "") and getattr(hf_client, "api_key", None) in (None, "")):
        if token:
            try:
                hf_client = InferenceClient(provider="hf-inference", api_key=token)
            except Exception as e:
                hf_client = None
        else:
            return default_result
    
    try:
        result = hf_client.text_classification(
            text,
            model="michellejieli/emotion_text_classifier",
        )
        
        # Store all emotions with their scores
        all_emotions = []
        for item in result:
            label = item.get('label', 'unknown').lower()
            conf = item.get('score', 0)
            mapped_score = EMOTION_SCORES.get(label, 3.0)
            all_emotions.append({
                'label': label,
                'confidence': conf,
                'mapped_score': mapped_score
            })
        
        if result and len(result) > 0:
            # Get the top emotion (highest confidence)
            top_emotion = result[0]
            emotion_label = top_emotion.get('label', 'neutral').lower()
            confidence = top_emotion.get('score', 0)
            emotion_score = EMOTION_SCORES.get(emotion_label, 3.0)
            
            return {
                'score': emotion_score,
                'label': emotion_label,
                'confidence': confidence,
                'all_emotions': all_emotions
            }
    except Exception as e:
        pass  # Emotion analysis failed, return default
    
    return default_result


def calculate_review_credibility(user_ratings_avg, emotion_result):
    """
    Calculate review credibility based on rating-emotion consistency.
    
    Returns:
        credibility_score (0-1): How credible the review appears
        flag (str): 'genuine', 'suspicious', or 'inconsistent'
        adjustment (float): Score adjustment to apply (-0.5 to +0.5)
    """
    emotion_label = emotion_result.get('label', 'neutral')
    emotion_score = emotion_result.get('score', 3.0)
    confidence = emotion_result.get('confidence', 0.0)
    
    # Determine emotion category
    if emotion_label in POSITIVE_EMOTIONS:
        emotion_category = 'positive'
    elif emotion_label in NEGATIVE_EMOTIONS:
        emotion_category = 'negative'
    else:
        emotion_category = 'neutral'
    
    # Determine rating category (based on user's average rating)
    if user_ratings_avg >= 4.0:
        rating_category = 'positive'
    elif user_ratings_avg <= 2.5:
        rating_category = 'negative'
    else:
        rating_category = 'neutral'
    
    # Check consistency
    # CONSISTENT: positive ratings + positive emotion, or negative ratings + negative emotion
    # INCONSISTENT: positive ratings + negative emotion (suspicious), or negative ratings + positive emotion (suspicious)
    
    if rating_category == emotion_category:
        # Consistent - genuine review
        credibility = 0.9 + (confidence * 0.1)  # 0.9 to 1.0
        flag = 'genuine'
        adjustment = confidence * 0.3  # Small bonus for consistent reviews
        
    elif emotion_category == 'neutral':
        # Neutral emotion with any rating - acceptable
        credibility = 0.7 + (confidence * 0.2)  # 0.7 to 0.9
        flag = 'genuine'
        adjustment = 0.0
        
    elif rating_category == 'neutral':
        # Neutral rating with positive/negative emotion - acceptable
        credibility = 0.75
        flag = 'genuine'
        adjustment = (emotion_score - 3.0) * 0.1  # Slight adjustment based on emotion
        
    else:
        # INCONSISTENT: positive ratings + negative emotion OR negative ratings + positive emotion
        credibility = 0.4 - (confidence * 0.2)  # 0.2 to 0.4 (lower credibility for more confident mismatch)
        flag = 'suspicious'
        
        if rating_category == 'positive' and emotion_category == 'negative':
            # User gave high stars but wrote negative text - possibly fake positive review
            adjustment = -0.5 * confidence  # Penalize more for confident negative emotion
        else:
            # User gave low stars but wrote positive text - unusual but less concerning
            adjustment = 0.3 * confidence  # Slight bonus - they might just be constructive
    
    return {
        'credibility': min(1.0, max(0.0, credibility)),
        'flag': flag,
        'adjustment': adjustment,
        'emotion_category': emotion_category,
        'rating_category': rating_category
    }


# Models
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(128))
    product_name = db.Column(db.String(128), nullable=False)
    batch_number = db.Column(db.String(64))
    category = db.Column(db.String(64))
    processor_score = db.Column(db.Float)
    ram_gb = db.Column(db.Integer)
    storage_gb = db.Column(db.Integer)
    battery_mah = db.Column(db.Integer)
    screen_inches = db.Column(db.Float)
    camera_mp = db.Column(db.Float)
    price_usd = db.Column(db.Float)
    weight_g = db.Column(db.Float)
    ideal_score = db.Column(db.Float, nullable=True)
    qr_code_path = db.Column(db.String(256), nullable=True)
    
    # Component Details - Processor
    processor_model = db.Column(db.String(128), nullable=True)
    
    # Component Details - RAM
    ram_type = db.Column(db.String(128), nullable=True)
    
    # Component Details - Storage
    storage_type = db.Column(db.String(128), nullable=True)
    
    # Component Details - Camera Sensors
    camera_sensor_main = db.Column(db.String(128), nullable=True)
    camera_sensor_ultra = db.Column(db.String(128), nullable=True)
    camera_sensor_telephoto = db.Column(db.String(128), nullable=True)
    
    # Component Details - Battery
    battery_tech = db.Column(db.String(128), nullable=True)
    charging_watt = db.Column(db.Float, nullable=True)
    
    # Component Details - Display
    display_type = db.Column(db.String(128), nullable=True)
    refresh_rate_hz = db.Column(db.Integer, nullable=True)
    
    # Blockchain Score Registry
    blockchain_hash = db.Column(db.String(128), nullable=True)
    blockchain_tx = db.Column(db.String(128), nullable=True)
    blockchain_timestamp = db.Column(db.String(64), nullable=True)
    blockchain_network = db.Column(db.String(32), nullable=True)
    
    # Redirect Links (where to buy)
    link_amazon = db.Column(db.String(512), nullable=True)
    link_flipkart = db.Column(db.String(512), nullable=True)
    link_official = db.Column(db.String(512), nullable=True)
    link_other = db.Column(db.String(512), nullable=True)
    
    # Relationship with reviews
    reviews = db.relationship('Review', backref='product', lazy=True, cascade='all, delete-orphan')
    
    @property
    def average_user_rating(self):
        """Calculate average user rating from all reviews."""
        if not self.reviews:
            return None
        total = sum(r.average_rating for r in self.reviews)
        return round(total / len(self.reviews), 2)


class Review(db.Model):
    """User reviews and ratings for products."""
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Link to user account
    
    # Individual ratings (1-5 stars)
    display_rating = db.Column(db.Float, nullable=False)
    performance_rating = db.Column(db.Float, nullable=False)
    battery_rating = db.Column(db.Float, nullable=False)
    camera_rating = db.Column(db.Float, nullable=False)
    design_rating = db.Column(db.Float, nullable=False)
    
    # Calculated average (adjusted based on credibility)
    average_rating = db.Column(db.Float, nullable=False)
    
    # Text review and emotion analysis
    review_text = db.Column(db.Text, nullable=True)
    user_emotion = db.Column(db.Float, nullable=True)       # Emotion score (1-5)
    emotion_label = db.Column(db.String(32), nullable=True) # joy, anger, sadness, etc.
    emotion_confidence = db.Column(db.Float, nullable=True) # Model confidence (0-1)
    
    # Credibility analysis
    credibility_score = db.Column(db.Float, nullable=True)  # 0-1 credibility score
    credibility_flag = db.Column(db.String(32), nullable=True)  # 'genuine', 'suspicious', etc.
    
    # Metadata
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, onupdate=db.func.current_timestamp())  # Track edits
    
    # Relationship to User
    user = db.relationship('User', backref=db.backref('reviews', lazy=True))
    
    # Unique constraint: one review per user per product
    __table_args__ = (db.UniqueConstraint('product_id', 'user_id', name='unique_user_product_review'),)
    
    def __repr__(self):
        return f'<Review {self.id} for Product {self.product_id} by User {self.user_id}>'


class PriceHistory(db.Model):
    """Track price changes over time for each product."""
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    price_usd = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    # Relationship
    product = db.relationship('Product', backref=db.backref('price_history', lazy=True, order_by='PriceHistory.recorded_at'))
    
    def __repr__(self):
        return f'<PriceHistory ${self.price_usd} for Product {self.product_id}>'


class PriceAlert(db.Model):
    """User alerts for price drops."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    target_price = db.Column(db.Float, nullable=False)  # Alert when price drops below this
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    triggered_at = db.Column(db.DateTime, nullable=True)  # When alert was triggered
    
    # Push notification subscription (for PWA)
    push_subscription = db.Column(db.Text, nullable=True)  # JSON subscription object
    
    # Relationships
    user = db.relationship('User', backref=db.backref('price_alerts', lazy=True))
    product = db.relationship('Product', backref=db.backref('alerts', lazy=True))
    
    # Unique constraint: one alert per user per product
    __table_args__ = (db.UniqueConstraint('user_id', 'product_id', name='unique_user_product_alert'),)
    
    def __repr__(self):
        return f'<PriceAlert ${self.target_price} for Product {self.product_id} by User {self.user_id}>'


class ChatMessage(db.Model):
    """Store chat history for the AI chatbot."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Nullable for anonymous users
    session_id = db.Column(db.String(64), nullable=False)  # For tracking conversation
    role = db.Column(db.String(16), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    # Relationship
    user = db.relationship('User', backref=db.backref('chat_messages', lazy=True))
    
    def __repr__(self):
        return f'<ChatMessage {self.role}: {self.content[:30]}...>'


# Create tables if they don't exist
with app.app_context():
    db.create_all()

# Templates have been moved to the `templates/` directory:
# - templates/base.html (shared layout)
# - templates/home.html
# - templates/producer_dashboard.html
# - templates/producer_register.html
# - templates/consumer_dashboard.html
# - templates/consumer_search.html
# - templates/consumer_scan.html
# Use `render_template('<name>.html')` in the route handlers below.

# ==========================================
# 3. ROUTES
# ==========================================

# ==========================================
# AUTHENTICATION ROUTES
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.role == 'producer':
            return redirect(url_for('producer_dashboard'))
        else:
            return redirect(url_for('consumer_dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user_type = request.form.get('user_type', 'consumer')  # 'consumer' or 'producer'
        remember_me = request.form.get('remember_me') == 'on'  # Checkbox value
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            if user.role == user_type:
                # Log in with remember option
                login_user(user, remember=remember_me)
                
                # Set session variables for timeout tracking
                session['last_activity'] = datetime.now().isoformat()
                session['remember_me'] = remember_me
                session.permanent = True  # Use PERMANENT_SESSION_LIFETIME
                
                flash(f'Welcome back, {username}!', 'success')
                if user.role == 'producer':
                    return redirect(url_for('producer_dashboard'))
                else:
                    return redirect(url_for('consumer_dashboard'))
            else:
                flash(f'This account is not a {user_type} account.', 'error')
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user_type = request.form.get('user_type', 'consumer')
        company_name = request.form.get('company_name', '').strip() if user_type == 'producer' else None
        
        # Validation
        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('signup'))
        
        if len(username) < 3 or len(username) > 15:
            flash('Username must be 3-15 characters.', 'error')
            return redirect(url_for('signup'))
        
        if len(password) < 5:
            flash('Password must be at least 5 characters.', 'error')
            return redirect(url_for('signup'))
        
        if user_type == 'producer' and not company_name:
            flash('Company name is required for producer accounts.', 'error')
            return redirect(url_for('signup'))
        
        # Check if username exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('signup'))
        
        # Create user
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username,
            password=hashed_password,
            role=user_type,
            company_name=company_name
        )
        db.session.add(new_user)
        db.session.commit()
        
        flash(f'Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  # Clear session data including timeout tracking
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if new_password:
            if len(new_password) < 5:
                flash('Password must be at least 5 characters.', 'error')
            elif new_password != confirm_password:
                flash('Passwords do not match.', 'error')
            else:
                current_user.password = generate_password_hash(new_password)
                db.session.commit()
                flash('Password updated successfully!', 'success')
        
        return redirect(url_for('profile'))
    
    return render_template('profile.html')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/producer')
@login_required
def producer_dashboard():
    if current_user.role != 'producer':
        flash('Access denied. Producer account required.', 'error')
        return redirect(url_for('consumer_dashboard'))
    
    # Filter products by producer's company
    products = Product.query.filter_by(company_name=current_user.company_name).order_by(Product.id.desc()).all()
    return render_template('producer_dashboard.html', products=products)


@app.route('/admin/reload-hf-client')
def reload_hf_client():
    """Reload HF client from current environment variable (useful after setting HF_TOKEN)."""
    global hf_client
    token = os.environ.get('HF_TOKEN')
    if not token:
        return "HF_TOKEN not found in environment for this process.", 400
    try:
        hf_client = InferenceClient(provider="hf-inference", api_key=token)
        return "HF client reloaded successfully.", 200
    except Exception as e:
        return f"Failed to reload HF client: {e}", 500


@app.route('/admin/set-hf-token', methods=['POST'])
def set_hf_token():
    """Set HF token at runtime (development use only) and initialize client."""
    global hf_client
    token = None
    if request.is_json:
        token = request.json.get('hf_token')
    else:
        token = request.form.get('hf_token')
    if not token:
        return "hf_token missing in request", 400
    os.environ['HF_TOKEN'] = token
    try:
        hf_client = InferenceClient(provider="hf-inference", api_key=token)
        return "HF token set and client initialized.", 200
    except Exception as e:
        return f"Failed to initialize HF client: {e}", 500

@app.route('/producer/product/<int:product_id>')
@login_required
def producer_product_detail(product_id):
    if current_user.role != 'producer':
        flash('Access denied. Producer account required.', 'error')
        return redirect(url_for('consumer_dashboard'))
    
    product = Product.query.get_or_404(product_id)
    
    # Check if product belongs to producer's company
    if product.company_name != current_user.company_name:
        flash('Access denied. This product does not belong to your company.', 'error')
        return redirect(url_for('producer_dashboard'))
    
    reviews = Review.query.filter_by(product_id=product_id).order_by(Review.timestamp.desc()).all()
    return render_template('producer_product_detail.html', product=product, reviews=reviews)


@app.route('/producer/product/<int:product_id>/edit-price', methods=['GET', 'POST'])
@login_required
def producer_edit_price(product_id):
    """Allow producers to edit only the price of their products."""
    if current_user.role != 'producer':
        flash('Access denied. Producer account required.', 'error')
        return redirect(url_for('consumer_dashboard'))
    
    product = Product.query.get_or_404(product_id)
    
    # Check if product belongs to producer's company
    if product.company_name != current_user.company_name:
        flash('Access denied. This product does not belong to your company.', 'error')
        return redirect(url_for('producer_dashboard'))
    
    if request.method == 'POST':
        new_price = _to_float(request.form.get('price_usd'))
        
        if new_price is None or new_price <= 0:
            flash('Please enter a valid price.', 'error')
            return redirect(url_for('producer_edit_price', product_id=product_id))
        
        old_price = product.price_usd
        
        # Check if price actually changed
        if new_price != old_price:
            product.price_usd = new_price
            
            # Recalculate the ideal score with new price
            product_dict = {
                'processor_score': product.processor_score,
                'ram_gb': product.ram_gb,
                'storage_gb': product.storage_gb,
                'battery_mah': product.battery_mah,
                'screen_inches': product.screen_inches,
                'camera_mp': product.camera_mp,
                'price_usd': new_price,
                'weight_g': product.weight_g,
            }
            new_score = score_product_from_db(product_dict, db.session)
            if new_score is not None:
                product.ideal_score = new_score
            
            # Invalidate blockchain verification (price changed = hash mismatch)
            if product.blockchain_hash:
                product.blockchain_hash = None
                product.blockchain_tx = None
                product.blockchain_timestamp = None
                product.blockchain_network = None
                flash(f'Price updated from ${old_price:.2f} to ${new_price:.2f}. Blockchain verification has been reset - please re-verify if needed.', 'warning')
            else:
                flash(f'Price updated from ${old_price:.2f} to ${new_price:.2f}. Score recalculated to {new_score:.1f}.', 'success')
            
            db.session.commit()
        else:
            flash('Price unchanged.', 'info')
        
        return redirect(url_for('producer_product_detail', product_id=product_id))
    
    return render_template('producer_edit_price.html', product=product)


@app.route('/producer/product/<int:product_id>/edit-links', methods=['GET', 'POST'])
@login_required
def producer_edit_links(product_id):
    """Allow producers to edit redirect links for their products."""
    if current_user.role != 'producer':
        flash('Access denied. Producer account required.', 'error')
        return redirect(url_for('consumer_dashboard'))
    
    product = Product.query.get_or_404(product_id)
    
    # Check if product belongs to producer's company
    if product.company_name != current_user.company_name:
        flash('Access denied. This product does not belong to your company.', 'error')
        return redirect(url_for('producer_dashboard'))
    
    if request.method == 'POST':
        product.link_amazon = request.form.get('link_amazon') or None
        product.link_flipkart = request.form.get('link_flipkart') or None
        product.link_official = request.form.get('link_official') or None
        product.link_other = request.form.get('link_other') or None
        
        db.session.commit()
        flash('Redirect links updated successfully.', 'success')
        return redirect(url_for('producer_product_detail', product_id=product_id))
    
    return render_template('producer_edit_links.html', product=product)


@app.route('/producer/product/<int:product_id>/delete', methods=['POST'])
@login_required
def producer_delete_product(product_id):
    """Allow producers to delete their products."""
    if current_user.role != 'producer':
        flash('Access denied. Producer account required.', 'error')
        return redirect(url_for('consumer_dashboard'))
    
    product = Product.query.get_or_404(product_id)
    
    # Check if product belongs to producer's company
    if product.company_name != current_user.company_name:
        flash('Access denied. This product does not belong to your company.', 'error')
        return redirect(url_for('producer_dashboard'))
    
    product_name = product.product_name
    
    # Delete the product (reviews cascade delete automatically)
    db.session.delete(product)
    db.session.commit()
    
    flash(f'Product "{product_name}" has been deleted.', 'success')
    return redirect(url_for('producer_dashboard'))


@app.route('/producer/register', methods=['GET', 'POST'])
@login_required
def producer_register():
    if current_user.role != 'producer':
        flash('Access denied. Producer account required.', 'error')
        return redirect(url_for('consumer_dashboard'))
    
    if request.method == 'POST':
        # Collect data from form - use producer's company name
        form = request.form
        p = Product(
            company_name = current_user.company_name,  # Use logged-in producer's company
            product_name = form.get('product_name'),
            batch_number = form.get('batch_number'),
            category = form.get('category'),
            processor_score = _to_float(form.get('processor_score')),
            ram_gb = _to_int(form.get('ram_gb')),
            storage_gb = _to_int(form.get('storage_gb')),
            battery_mah = _to_int(form.get('battery_mah')),
            screen_inches = _to_float(form.get('screen_inches')),
            camera_mp = _to_float(form.get('camera_mp')),
            price_usd = _to_float(form.get('price_usd')),
            weight_g = _to_float(form.get('weight_g')),
            # Component details
            processor_model = form.get('processor_model'),
            ram_type = form.get('ram_type'),
            storage_type = form.get('storage_type'),
            camera_sensor_main = form.get('camera_sensor_main'),
            camera_sensor_ultra = form.get('camera_sensor_ultra'),
            camera_sensor_telephoto = form.get('camera_sensor_telephoto'),
            battery_tech = form.get('battery_tech'),
            charging_watt = _to_float(form.get('charging_watt')),
            display_type = form.get('display_type'),
            refresh_rate_hz = _to_int(form.get('refresh_rate_hz')),
            # Redirect links
            link_amazon = form.get('link_amazon') or None,
            link_flipkart = form.get('link_flipkart') or None,
            link_official = form.get('link_official') or None,
            link_other = form.get('link_other') or None,
        )
        db.session.add(p)
        db.session.commit()
        
        # Calculate ideal score
        product_dict = {
            'processor_score': p.processor_score,
            'ram_gb': p.ram_gb,
            'storage_gb': p.storage_gb,
            'battery_mah': p.battery_mah,
            'screen_inches': p.screen_inches,
            'camera_mp': p.camera_mp,
            'price_usd': p.price_usd,
            'weight_g': p.weight_g,
        }
        ideal_score = score_product_from_db(product_dict, db.session)
        if ideal_score is not None:
            p.ideal_score = ideal_score
        
        # Generate QR code
        qr_path = _generate_qr_code(p)
        p.qr_code_path = qr_path
        db.session.commit()
        
        score_msg = f" Score: {ideal_score:.1f}" if ideal_score else ""
        flash(f"Product '{p.product_name}' registered.{score_msg}")
        return redirect(url_for('producer_dashboard'))
    return render_template('producer_register.html')


def _to_int(v):
    try:
        return int(v) if v not in (None, '') else None
    except Exception:
        return None


def _to_float(v):
    try:
        return float(v) if v not in (None, '') else None
    except Exception:
        return None


def _generate_qr_code(product):
    """Generate QR code for a product and return the file path."""
    # Create static/qr_codes directory if it doesn't exist
    qr_dir = os.path.join(BASE_DIR, 'static', 'qr_codes')
    os.makedirs(qr_dir, exist_ok=True)
    
    # QR code content: URL to product details (or product ID)
    qr_content = f"https://credilens.app/product/{product.id}"
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save to file
    filename = f"product_{product.id}.png"
    filepath = os.path.join(qr_dir, filename)
    img.save(filepath)
    
    return f"static/qr_codes/{filename}"


@app.route('/qr/<int:product_id>')
def view_qr(product_id):
    """Display QR code for a product."""
    product = Product.query.get_or_404(product_id)
    if not product.qr_code_path:
        return "QR code not found", 404
    qr_full_path = os.path.join(BASE_DIR, product.qr_code_path)
    return send_file(qr_full_path, mimetype='image/png')


@app.route('/consumer')
@login_required
def consumer_dashboard():
    if current_user.role != 'consumer':
        flash('Access denied. Consumer account required.', 'error')
        return redirect(url_for('producer_dashboard'))
    
    products = Product.query.filter(Product.ideal_score.isnot(None)).order_by(Product.ideal_score.desc()).all()
    return render_template('consumer_dashboard.html', products=products)

@app.route('/consumer/search')
@login_required
def consumer_search():
    if current_user.role != 'consumer':
        flash('Access denied. Consumer account required.', 'error')
        return redirect(url_for('producer_dashboard'))
    return render_template('consumer_search.html')

@app.route('/consumer/scan')
@login_required
def consumer_scan():
    if current_user.role != 'consumer':
        flash('Access denied. Consumer account required.', 'error')
        return redirect(url_for('producer_dashboard'))
    return render_template('consumer_scan.html')


@app.route('/consumer/ai-scan')
@login_required
def consumer_ai_scan():
    """AI-powered phone scanner page."""
    if current_user.role != 'consumer':
        flash('Access denied. Consumer account required.', 'error')
        return redirect(url_for('producer_dashboard'))
    return render_template('consumer_ai_scan.html')


@app.route('/api/identify-phone', methods=['POST'])
@login_required
def api_identify_phone():
    """
    API endpoint to identify a phone from an image using Google Gemini.
    Expects JSON with 'image' field containing base64 encoded image.
    """
    if current_user.role != 'consumer':
        return jsonify({"error": "Access denied"}), 403
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Get base64 image data (remove data URL prefix if present)
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Call Gemini API to identify the phone
        result = identify_phone_with_gemini(image_data)
        
        if "error" in result:
            return jsonify(result), 500
        
        # If phone was identified, try to match in database
        if result.get("identified"):
            brand = result.get("brand")
            model = result.get("model")
            
            matched_product = match_phone_in_database(brand, model)
            
            if matched_product:
                result["matched"] = True
                result["product"] = {
                    "id": matched_product.id,
                    "name": matched_product.product_name,
                    "company": matched_product.company_name,
                    "ideal_score": matched_product.ideal_score,
                    "price_usd": matched_product.price_usd,
                    "processor_score": matched_product.processor_score,
                    "ram_gb": matched_product.ram_gb,
                    "storage_gb": matched_product.storage_gb,
                    "battery_mah": matched_product.battery_mah,
                    "camera_mp": matched_product.camera_mp,
                    "average_user_rating": matched_product.average_user_rating,
                    "url": url_for('consumer_product_detail', product_id=matched_product.id)
                }
            else:
                result["matched"] = False
                result["product"] = None
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/consumer/product/<int:product_id>')
@login_required
def consumer_product_detail(product_id):
    if current_user.role != 'consumer':
        flash('Access denied. Consumer account required.', 'error')
        return redirect(url_for('producer_dashboard'))
    
    product = Product.query.get_or_404(product_id)
    reviews = Review.query.filter_by(product_id=product_id).order_by(Review.timestamp.desc()).all()
    
    # Check if current user already has a review for this product
    user_review = Review.query.filter_by(product_id=product_id, user_id=current_user.id).first()
    
    return render_template('consumer_product_detail.html', product=product, reviews=reviews, user_review=user_review)

@app.route('/consumer/product/<int:product_id>/review', methods=['POST'])
@login_required
def submit_review(product_id):
    """Submit or update a user review for a product. One review per user per product."""
    if current_user.role != 'consumer':
        flash('Access denied. Consumer account required.', 'error')
        return redirect(url_for('producer_dashboard'))
    
    product = Product.query.get_or_404(product_id)
    
    try:
        # Get ratings from form
        display = float(request.form.get('display_rating', 0))
        performance = float(request.form.get('performance_rating', 0))
        battery = float(request.form.get('battery_rating', 0))
        camera = float(request.form.get('camera_rating', 0))
        design = float(request.form.get('design_rating', 0))
        
        # Get review text (mandatory)
        review_text = request.form.get('review_text', '').strip()
        if not review_text:
            flash('Please write a review (even one word is fine).', 'error')
            return redirect(url_for('consumer_product_detail', product_id=product_id))
        
        # Validate ratings (1-5)
        ratings = [display, performance, battery, camera, design]
        if not all(1 <= r <= 5 for r in ratings):
            flash('All ratings must be between 1 and 5 stars.', 'error')
            return redirect(url_for('consumer_product_detail', product_id=product_id))
        
        # Calculate base average from user ratings ONLY (not emotion)
        user_ratings_avg = round(sum(ratings) / len(ratings), 2)
        
        # Analyze emotion from review text (always recalculate on submit/update)
        emotion_result = analyze_emotion(review_text)
        emotion_score = emotion_result.get('score', 3.0)
        emotion_label = emotion_result.get('label', 'neutral')
        emotion_confidence = emotion_result.get('confidence', 0.0)
        
        # Calculate credibility and get adjustment
        credibility = calculate_review_credibility(user_ratings_avg, emotion_result)
        
        # Final average: User ratings with small emotion-based adjustment
        adjustment = credibility.get('adjustment', 0.0)
        final_average = round(min(5.0, max(1.0, user_ratings_avg + adjustment)), 2)
        
        # Check if user already has a review for this product
        existing_review = Review.query.filter_by(product_id=product_id, user_id=current_user.id).first()
        
        if existing_review:
            # UPDATE existing review
            existing_review.display_rating = display
            existing_review.performance_rating = performance
            existing_review.battery_rating = battery
            existing_review.camera_rating = camera
            existing_review.design_rating = design
            existing_review.average_rating = final_average
            existing_review.review_text = review_text
            existing_review.user_emotion = emotion_score
            existing_review.emotion_label = emotion_label
            existing_review.emotion_confidence = emotion_confidence
            existing_review.credibility_score = credibility.get('credibility', 0.7)
            existing_review.credibility_flag = credibility.get('flag', 'genuine')
            # updated_at is handled automatically by onupdate
            
            db.session.commit()
            
            # Generate feedback message
            if credibility['flag'] == 'genuine':
                flash(f'Review updated! Score: {final_average}/5 (Emotion: {emotion_label})', 'success')
            elif credibility['flag'] == 'suspicious':
                flash(f'Review updated. Score: {final_average}/5. Note: Your rating and text sentiment appear inconsistent.', 'warning')
            else:
                flash(f'Review updated! Score: {final_average}/5 (Emotion: {emotion_label})', 'success')
        else:
            # CREATE new review
            review = Review(
                product_id=product_id,
                user_id=current_user.id,
                display_rating=display,
                performance_rating=performance,
                battery_rating=battery,
                camera_rating=camera,
                design_rating=design,
                average_rating=final_average,
                review_text=review_text,
                user_emotion=emotion_score,
                emotion_label=emotion_label,
                emotion_confidence=emotion_confidence,
                credibility_score=credibility.get('credibility', 0.7),
                credibility_flag=credibility.get('flag', 'genuine')
            )
            
            db.session.add(review)
            db.session.commit()
            
            # Generate feedback message
            if credibility['flag'] == 'genuine':
                flash(f'Thank you for your review! Score: {final_average}/5 (Emotion: {emotion_label})', 'success')
            elif credibility['flag'] == 'suspicious':
                flash(f'Review submitted. Score: {final_average}/5. Note: Your rating and text sentiment appear inconsistent.', 'warning')
            else:
                flash(f'Review submitted! Score: {final_average}/5 (Emotion: {emotion_label})', 'success')
        
        return redirect(url_for('consumer_product_detail', product_id=product_id))
        
    except ValueError:
        flash('Invalid rating values. Please try again.', 'error')
        return redirect(url_for('consumer_product_detail', product_id=product_id))

@app.route('/consumer/custom-scores', methods=['GET', 'POST'])
@login_required
def consumer_custom_scores():
    if current_user.role != 'consumer':
        flash('Access denied. Consumer account required.', 'error')
        return redirect(url_for('producer_dashboard'))
    
    # Priority to weight mapping (1st priority = 8, 2nd = 7, etc.)
    priority_to_weight = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 4, 7: 3, 8: 2}
    
    # Default weights
    default_weights = {
        "processor_score": 8,
        "ram_gb": 7,
        "storage_gb": 6,
        "battery_mah": 5,
        "screen_inches": 4,
        "camera_mp": 4,
        "price_usd": 3,
        "weight_g": 2,
    }
    
    # Default priorities
    default_priorities = {
        'processor': 1,
        'ram': 2,
        'storage': 3,
        'battery': 4,
        'screen': 5,
        'camera': 6,
        'price': 7,
        'weight': 8
    }
    
    # Get custom weights from priorities or use defaults
    if request.method == 'POST':
        # Get priorities from form
        priorities = {
            'processor': int(request.form.get('processor_priority', 1)),
            'ram': int(request.form.get('ram_priority', 2)),
            'storage': int(request.form.get('storage_priority', 3)),
            'battery': int(request.form.get('battery_priority', 4)),
            'screen': int(request.form.get('screen_priority', 5)),
            'camera': int(request.form.get('camera_priority', 6)),
            'price': int(request.form.get('price_priority', 7)),
            'weight': int(request.form.get('weight_priority', 8))
        }
        
        # Convert priorities to weights
        custom_weights = {
            "processor_score": priority_to_weight[priorities['processor']],
            "ram_gb": priority_to_weight[priorities['ram']],
            "storage_gb": priority_to_weight[priorities['storage']],
            "battery_mah": priority_to_weight[priorities['battery']],
            "screen_inches": priority_to_weight[priorities['screen']],
            "camera_mp": priority_to_weight[priorities['camera']],
            "price_usd": priority_to_weight[priorities['price']],
            "weight_g": priority_to_weight[priorities['weight']],
        }
    else:
        custom_weights = default_weights
        priorities = default_priorities
    
    # Calculate custom scores
    products = calculate_custom_scores_from_db(db.session, custom_weights)
    
    # Sort by custom score descending
    products_sorted = sorted([p for p in products if p.get('custom_score') is not None], 
                            key=lambda x: x['custom_score'], reverse=True)
    
    return render_template('consumer_custom_scores.html', 
                          products=products_sorted, 
                          weights=custom_weights,
                          priorities=priorities)


# ==========================================
# PRODUCT COMPARISON DASHBOARD
# ==========================================

@app.route('/consumer/compare')
@login_required
def consumer_compare():
    """Product comparison dashboard - compare up to 3 products side by side."""
    if current_user.role != 'consumer':
        flash('Access denied. Consumer account required.', 'error')
        return redirect(url_for('producer_dashboard'))
    
    # Get product IDs from query params (e.g., /compare?ids=1,2,3)
    ids_param = request.args.get('ids', '')
    
    selected_products = []
    if ids_param:
        try:
            product_ids = [int(id.strip()) for id in ids_param.split(',') if id.strip()][:3]  # Max 3
            selected_products = Product.query.filter(Product.id.in_(product_ids)).all()
        except ValueError:
            pass
    
    # Get all products for the selector
    all_products = Product.query.filter(Product.ideal_score.isnot(None)).order_by(Product.product_name).all()
    
    # Calculate comparison data
    comparison_data = []
    for product in selected_products:
        # Get review stats
        reviews = Review.query.filter_by(product_id=product.id).all()
        avg_rating = sum(r.rating for r in reviews) / len(reviews) if reviews else 0
        review_count = len(reviews)
        
        # Calculate credibility percentage
        credible_reviews = [r for r in reviews if r.credibility_flag == 'genuine']
        credibility_pct = (len(credible_reviews) / len(reviews) * 100) if reviews else 0
        
        comparison_data.append({
            'product': product,
            'avg_rating': round(avg_rating, 1),
            'review_count': review_count,
            'credibility_pct': round(credibility_pct, 0),
        })
    
    return render_template('consumer_compare.html',
                          comparison_data=comparison_data,
                          all_products=all_products,
                          selected_ids=[p.id for p in selected_products])


@app.route('/api/products/search')
@login_required
def api_product_search():
    """API endpoint for product search (used by comparison selector)."""
    query = request.args.get('q', '').strip().lower()
    
    if not query or len(query) < 2:
        return jsonify([])
    
    products = Product.query.filter(
        Product.ideal_score.isnot(None),
        (Product.product_name.ilike(f'%{query}%') | Product.company_name.ilike(f'%{query}%'))
    ).limit(10).all()
    
    return jsonify([{
        'id': p.id,
        'name': p.product_name,
        'company': p.company_name,
        'score': p.ideal_score
    } for p in products])


# ==========================================
# 5. BLOCKCHAIN SCORE REGISTRY
# ==========================================

@app.route('/api/blockchain/status')
def blockchain_status():
    """Check blockchain connection status."""
    status = blockchain.get_blockchain_status()
    return jsonify(status)


@app.route('/api/blockchain/register/<int:product_id>', methods=['POST'])
@login_required
def register_on_blockchain(product_id):
    """Register a product's score on the blockchain."""
    if current_user.role != 'producer':
        return jsonify({"error": "Only producers can register scores on blockchain"}), 403
    
    product = Product.query.get_or_404(product_id)
    
    # Check if already registered
    if product.blockchain_tx:
        return jsonify({
            "error": "Already registered",
            "tx_hash": product.blockchain_tx,
            "message": "This product is already on the blockchain"
        }), 400
    
    # Create specs dictionary for hashing
    specs = {
        "processor_score": product.processor_score,
        "ram_gb": product.ram_gb,
        "storage_gb": product.storage_gb,
        "battery_mah": product.battery_mah,
        "camera_mp": product.camera_mp,
        "price_usd": product.price_usd
    }
    
    # Create hash
    hash_result = blockchain.create_score_hash(
        product_id=product.id,
        product_name=product.product_name,
        score=product.ideal_score,
        specs_dict=specs
    )
    
    # Store on blockchain
    tx_result = blockchain.store_hash_on_blockchain(hash_result["hash"])
    
    if tx_result["success"]:
        # Save to database
        product.blockchain_hash = hash_result["hash"]
        product.blockchain_tx = tx_result["tx_hash"]
        product.blockchain_timestamp = hash_result["timestamp"]
        product.blockchain_network = tx_result["network"]
        db.session.commit()
        
        explorer_url = blockchain.get_blockchain_explorer_url(
            tx_result["tx_hash"], 
            tx_result["network"]
        )
        
        return jsonify({
            "success": True,
            "tx_hash": tx_result["tx_hash"],
            "hash": hash_result["hash"],
            "network": tx_result["network"],
            "explorer_url": explorer_url,
            "message": tx_result["message"]
        })
    else:
        return jsonify({
            "success": False,
            "error": tx_result.get("error", "Unknown error")
        }), 500


@app.route('/api/blockchain/verify/<int:product_id>')
def verify_blockchain_score(product_id):
    """Verify a product's score against blockchain record."""
    product = Product.query.get_or_404(product_id)
    
    if not product.blockchain_hash:
        return jsonify({
            "verified": False,
            "error": "Product not registered on blockchain"
        })
    
    # Create specs dictionary
    specs = {
        "processor_score": product.processor_score,
        "ram_gb": product.ram_gb,
        "storage_gb": product.storage_gb,
        "battery_mah": product.battery_mah,
        "camera_mp": product.camera_mp,
        "price_usd": product.price_usd
    }
    
    # Verify hash
    result = blockchain.verify_score_hash(
        product_id=product.id,
        product_name=product.product_name,
        claimed_score=product.ideal_score,
        specs_dict=specs,
        stored_hash=product.blockchain_hash,
        stored_timestamp=product.blockchain_timestamp
    )
    
    explorer_url = blockchain.get_blockchain_explorer_url(
        product.blockchain_tx,
        product.blockchain_network
    )
    
    return jsonify({
        "verified": result["verified"],
        "product_name": product.product_name,
        "score": product.ideal_score,
        "tx_hash": product.blockchain_tx,
        "network": product.blockchain_network,
        "explorer_url": explorer_url,
        "timestamp": product.blockchain_timestamp
    })


# ==========================================
# DATABASE SEED ROUTE (for PostgreSQL on Render)
# ==========================================
@app.route('/admin/seed-database', methods=['GET', 'POST'])
def seed_database():
    """
    Seed the database with initial products and users.
    Protected by a secret key to prevent unauthorized access.
    Visit: /admin/seed-database?key=YOUR_SECRET_KEY
    """
    # Use SECRET_KEY or a dedicated SEED_KEY for protection
    seed_key = os.environ.get('SEED_KEY', 'credilens-seed-2025')
    provided_key = request.args.get('key', '')
    
    if provided_key != seed_key:
        return jsonify({
            "error": "Unauthorized. Provide correct ?key= parameter.",
            "hint": "Set SEED_KEY environment variable on Render"
        }), 403
    
    try:
        # Check if database already has data
        existing_products = Product.query.count()
        existing_users = User.query.count()
        
        if existing_products > 0 or existing_users > 0:
            if request.method != 'POST' and request.args.get('force') != 'true':
                return jsonify({
                    "status": "Database already has data",
                    "products": existing_products,
                    "users": existing_users,
                    "action": "Add ?force=true to clear and reseed, or POST to this URL"
                })
            
            # Clear existing data if force=true
            if request.args.get('force') == 'true':
                Review.query.delete()
                Product.query.delete()
                User.query.delete()
                db.session.commit()
        
        # Seed users
        users_created = 0
        for user_data in SEED_USERS:
            if not User.query.filter_by(username=user_data['username']).first():
                user = User(
                    username=user_data['username'],
                    password=generate_password_hash(user_data['password']),
                    role=user_data['role'],
                    company_name=user_data.get('company_name')
                )
                db.session.add(user)
                users_created += 1
        
        db.session.commit()
        
        # Seed products
        products_created = 0
        for p_data in SEED_PRODUCTS:
            # Generate search links for e-commerce sites
            links = generate_search_links(p_data['product_name'], p_data['company_name'])
            
            product = Product(
                company_name=p_data['company_name'],
                product_name=p_data['product_name'],
                batch_number=p_data['batch_number'],
                category=p_data['category'],
                processor_score=p_data['processor_score'],
                ram_gb=p_data['ram_gb'],
                storage_gb=p_data['storage_gb'],
                battery_mah=p_data['battery_mah'],
                screen_inches=p_data['screen_inches'],
                camera_mp=p_data['camera_mp'],
                price_usd=p_data['price_usd'],
                weight_g=p_data['weight_g'],
                ideal_score=p_data['ideal_score'],
                processor_model=p_data.get('processor_model'),
                ram_type=p_data.get('ram_type'),
                storage_type=p_data.get('storage_type'),
                camera_sensor_main=p_data.get('camera_sensor_main'),
                camera_sensor_ultra=p_data.get('camera_sensor_ultra'),
                camera_sensor_telephoto=p_data.get('camera_sensor_telephoto'),
                battery_tech=p_data.get('battery_tech'),
                charging_watt=p_data.get('charging_watt'),
                display_type=p_data.get('display_type'),
                refresh_rate_hz=p_data.get('refresh_rate_hz'),
                link_amazon=links['link_amazon'],
                link_flipkart=links['link_flipkart'],
                link_official=links['link_official'],
                link_other=links['link_other'],
            )
            db.session.add(product)
            products_created += 1
        
        db.session.commit()
        
        return jsonify({
            "status": "success",
            "message": "Database seeded successfully!",
            "products_created": products_created,
            "users_created": users_created,
            "total_products": Product.query.count(),
            "total_users": User.query.count()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ==========================================
# PRICE TRACKING & ALERTS
# ==========================================

@app.route('/api/product/<int:product_id>/price-history')
def get_price_history(product_id):
    """Get price history for a product (for chart)."""
    product = Product.query.get_or_404(product_id)
    history = PriceHistory.query.filter_by(product_id=product_id).order_by(PriceHistory.recorded_at).all()
    
    # If no history, return current price as single point
    if not history:
        return jsonify({
            'product_name': product.product_name,
            'current_price': product.price_usd,
            'history': [{
                'price': product.price_usd,
                'date': datetime.now().isoformat()
            }]
        })
    
    return jsonify({
        'product_name': product.product_name,
        'current_price': product.price_usd,
        'history': [{'price': h.price_usd, 'date': h.recorded_at.isoformat()} for h in history]
    })


@app.route('/consumer/product/<int:product_id>/set-alert', methods=['POST'])
@login_required
def set_price_alert(product_id):
    """Set a price drop alert for a product."""
    if current_user.role != 'consumer':
        flash('Only consumers can set price alerts.', 'error')
        return redirect(url_for('consumer_product_detail', product_id=product_id))
    
    product = Product.query.get_or_404(product_id)
    target_price = request.form.get('target_price', type=float)
    
    if not target_price or target_price <= 0:
        flash('Please enter a valid target price.', 'error')
        return redirect(url_for('consumer_product_detail', product_id=product_id))
    
    if target_price >= product.price_usd:
        flash('Target price must be lower than current price.', 'error')
        return redirect(url_for('consumer_product_detail', product_id=product_id))
    
    # Check if alert already exists
    existing = PriceAlert.query.filter_by(user_id=current_user.id, product_id=product_id).first()
    
    if existing:
        existing.target_price = target_price
        existing.is_active = True
        existing.triggered_at = None
        flash(f'Price alert updated! We\'ll notify you when price drops below ${target_price:.2f}', 'success')
    else:
        alert = PriceAlert(
            user_id=current_user.id,
            product_id=product_id,
            target_price=target_price
        )
        db.session.add(alert)
        flash(f'Price alert set! We\'ll notify you when price drops below ${target_price:.2f}', 'success')
    
    db.session.commit()
    return redirect(url_for('consumer_product_detail', product_id=product_id))


@app.route('/consumer/my-alerts')
@login_required
def my_price_alerts():
    """View and manage price alerts."""
    if current_user.role != 'consumer':
        flash('Only consumers can view price alerts.', 'error')
        return redirect(url_for('home'))
    
    alerts = PriceAlert.query.filter_by(user_id=current_user.id).order_by(PriceAlert.created_at.desc()).all()
    return render_template('my_alerts.html', alerts=alerts)


@app.route('/consumer/alert/<int:alert_id>/delete', methods=['POST'])
@login_required
def delete_price_alert(alert_id):
    """Delete a price alert."""
    alert = PriceAlert.query.get_or_404(alert_id)
    
    if alert.user_id != current_user.id:
        flash('Unauthorized.', 'error')
        return redirect(url_for('my_price_alerts'))
    
    db.session.delete(alert)
    db.session.commit()
    flash('Price alert deleted.', 'success')
    return redirect(url_for('my_price_alerts'))


# ==========================================
# EXCEL/CSV IMPORT FOR PRODUCTS
# ==========================================

@app.route('/admin/import-products', methods=['GET', 'POST'])
def import_products():
    """Import products from CSV file."""
    # Simple key-based auth
    import_key = os.environ.get('IMPORT_KEY', 'credilens-import-2025')
    provided_key = request.args.get('key', '')
    
    if provided_key != import_key:
        return jsonify({"error": "Unauthorized. Provide correct ?key= parameter."}), 403
    
    if request.method == 'GET':
        return '''
        <html>
        <head><title>Import Products - CrediLens</title>
        <style>body{font-family:Arial;max-width:600px;margin:40px auto;padding:20px}
        h1{color:#2563eb}form{background:#f3f4f6;padding:20px;border-radius:8px}
        input[type=file]{margin:10px 0}button{background:#2563eb;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer}</style>
        </head>
        <body>
        <h1>📱 Import Products</h1>
        <p>Upload a CSV file with product data. <a href="/static/product_import_template.csv">Download template</a></p>
        <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required><br>
        <button type="submit">Import Products</button>
        </form>
        </body></html>
        '''
    
    # POST - process the file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Read CSV
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        reader = csv.DictReader(stream)
        
        products_created = 0
        errors = []
        
        for row_num, row in enumerate(reader, start=2):
            try:
                # Required fields
                if not row.get('product_name') or not row.get('company_name'):
                    errors.append(f"Row {row_num}: Missing product_name or company_name")
                    continue
                
                # Create product
                product = Product(
                    product_name=row['product_name'].strip(),
                    company_name=row['company_name'].strip(),
                    batch_number=row.get('batch_number', f"IMPORT-{datetime.now().strftime('%Y%m%d')}-{row_num}"),
                    category=row.get('category', 'Smartphone'),
                    processor_score=float(row['processor_score']) if row.get('processor_score') else None,
                    ram_gb=float(row['ram_gb']) if row.get('ram_gb') else None,
                    storage_gb=float(row['storage_gb']) if row.get('storage_gb') else None,
                    battery_mah=float(row['battery_mah']) if row.get('battery_mah') else None,
                    screen_inches=float(row['screen_inches']) if row.get('screen_inches') else None,
                    camera_mp=float(row['camera_mp']) if row.get('camera_mp') else None,
                    price_usd=float(row['price_usd']) if row.get('price_usd') else None,
                    weight_g=float(row['weight_g']) if row.get('weight_g') else None,
                    processor_model=row.get('processor_model'),
                    ram_type=row.get('ram_type'),
                    storage_type=row.get('storage_type'),
                    camera_sensor_main=row.get('camera_sensor_main'),
                    camera_sensor_ultra=row.get('camera_sensor_ultra'),
                    camera_sensor_telephoto=row.get('camera_sensor_telephoto'),
                    battery_tech=row.get('battery_tech'),
                    charging_watt=float(row['charging_watt']) if row.get('charging_watt') else None,
                    display_type=row.get('display_type'),
                    refresh_rate_hz=float(row['refresh_rate_hz']) if row.get('refresh_rate_hz') else None,
                    link_amazon=row.get('link_amazon'),
                    link_flipkart=row.get('link_flipkart'),
                    link_official=row.get('link_official'),
                    link_other=row.get('link_other'),
                )
                
                # Calculate score
                db.session.add(product)
                db.session.flush()  # Get the ID
                
                score = score_product_from_db(product.id)
                if score:
                    product.ideal_score = score
                
                # Record initial price history
                price_record = PriceHistory(product_id=product.id, price_usd=product.price_usd)
                db.session.add(price_record)
                
                products_created += 1
                
            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")
        
        db.session.commit()
        
        return jsonify({
            "status": "success",
            "products_created": products_created,
            "errors": errors[:10] if errors else [],
            "total_errors": len(errors)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ==========================================
# AI CHATBOT
# ==========================================

def get_chatbot_context():
    """Get product database context for the chatbot."""
    products = Product.query.all()
    context = "You are CrediBot, the AI assistant for CrediLens - a smartphone credibility platform. "
    context += "You help users find information about phones, compare products, and understand credibility scores. "
    context += f"\n\nDatabase has {len(products)} products. Here are some:\n"
    
    for p in products[:20]:  # Limit to 20 for context
        context += f"- {p.product_name} by {p.company_name}: ${p.price_usd}, Score: {p.ideal_score}, "
        context += f"RAM: {p.ram_gb}GB, Storage: {p.storage_gb}GB, Battery: {p.battery_mah}mAh\n"
    
    context += "\nCredibility scores are 0-100 based on specs vs price. Higher is better value."
    return context


@app.route('/chatbot')
def chatbot_page():
    """Chatbot interface page."""
    # Generate session ID for anonymous users
    if 'chat_session_id' not in session:
        session['chat_session_id'] = str(uuid.uuid4())
    
    return render_template('chatbot.html')


@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint for chatbot messages."""
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    session_id = session.get('chat_session_id', str(uuid.uuid4()))
    
    # Save user message
    user_msg = ChatMessage(
        user_id=current_user.id if current_user.is_authenticated else None,
        session_id=session_id,
        role='user',
        content=user_message
    )
    db.session.add(user_msg)
    
    try:
        # Try Google Gemini first (free tier)
        response_text = call_gemini_chat(user_message)
        
        if not response_text:
            # Fallback to HuggingFace
            response_text = call_huggingface_chat(user_message)
        
        if not response_text:
            response_text = "I'm sorry, I couldn't process your request right now. Please try again."
        
        # Save assistant response
        assistant_msg = ChatMessage(
            user_id=current_user.id if current_user.is_authenticated else None,
            session_id=session_id,
            role='assistant',
            content=response_text
        )
        db.session.add(assistant_msg)
        db.session.commit()
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


def call_gemini_chat(user_message):
    """Call Google Gemini API for chat response."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    
    context = get_chatbot_context()
    
    body = {
        "contents": [{
            "parts": [{"text": f"{context}\n\nUser: {user_message}\n\nAssistant:"}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 500,
        }
    }
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    try:
        response = requests.post(url, json=body, timeout=15)
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
    except:
        pass
    
    return None


def call_huggingface_chat(user_message):
    """Fallback to HuggingFace Inference API."""
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not api_key:
        return None
    
    try:
        client = InferenceClient(token=api_key)
        context = get_chatbot_context()
        
        response = client.text_generation(
            f"{context}\n\nUser: {user_message}\n\nAssistant:",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_new_tokens=300
        )
        return response
    except:
        return None


# ==========================================
# 6. RUNNER
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, port=8000)