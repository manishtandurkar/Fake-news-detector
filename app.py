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
import traceback
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
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.