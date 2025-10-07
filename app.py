from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

class FakeNewsAPI:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove punctuation but keep spaces
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            if os.path.exists('fake_news_model.pkl') and os.path.exists('vectorizer.pkl'):
                with open('fake_news_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                with open('vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.is_loaded = True
                print("✅ Model loaded successfully!")
                return True
            else:
                print("❌ Model files not found. Please train the model first.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def train_model(self):
        """Train the model if not already loaded"""
        print("Training new model...")
        from main import FakeNewsDetector
        
        # Create and train detector
        detector = FakeNewsDetector()
        if detector.load_data() and detector.prepare_data():
            detector.train_model('logistic')
            detector.save_model()
            
            # Load the newly trained model
            self.model = detector.model
            self.vectorizer = detector.vectorizer
            self.is_loaded = True
            return True
        return False
    
    def predict(self, text):
        """Predict if text is fake or real news"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        if len(clean_text.strip()) == 0:
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0,
                'probabilities': {'fake': 50, 'real': 50},
                'error': 'Text too short after preprocessing'
            }
        
        try:
            # Vectorize
            text_tfidf = self.vectorizer.transform([clean_text])
            
            # Predict
            prediction = self.model.predict(text_tfidf)[0]
            probability = self.model.predict_proba(text_tfidf)[0]
            
            return {
                'prediction': 'FAKE' if prediction == 0 else 'REAL',
                'confidence': float(max(probability) * 100),
                'probabilities': {
                    'fake': float(probability[0] * 100),
                    'real': float(probability[1] * 100)
                },
                'preprocessed_text': clean_text[:100] + '...' if len(clean_text) > 100 else clean_text
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize the API
detector_api = FakeNewsAPI()

@app.route('/')
def home():
    return jsonify({"message": "Fake News Detection API is running!"})

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": detector_api.is_loaded
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) < 10:
            return jsonify({"error": "Text too short. Please provide at least 10 characters."}), 400
        
        # Make prediction
        result = detector_api.predict(text)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Endpoint to retrain the model"""
    try:
        success = detector_api.train_model()
        if success:
            return jsonify({
                "success": True,
                "message": "Model trained successfully!"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to train model"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("="*50)
    print("FAKE NEWS DETECTION API")
    print("="*50)
    
    # Try to load existing model
    if not detector_api.load_model():
        print("No existing model found. You'll need to train one first.")
        print("Visit /api/train to train a new model or run main.py first.")
    
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)