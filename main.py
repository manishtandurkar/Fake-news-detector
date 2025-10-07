import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import pickle
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDetector:
    def __init__(self, fake_path='ISOT/Fake.csv', real_path='ISOT/True.csv'):
        """Initialize the fake news detector with ISOT dataset paths"""
        self.fake_path = fake_path
        self.real_path = real_path
        self.df = None
        self.vectorizer = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load ISOT dataset from separate fake and real files"""
        print("Loading ISOT dataset from separate files...")
        try:
            # Load fake news
            print(f"Loading fake news from: {self.fake_path}")
            fake_df = pd.read_csv(self.fake_path)
            fake_df['label'] = 0  # 0 for fake
            print(f"Loaded {len(fake_df)} fake news articles")
            
            # Load real news  
            print(f"Loading real news from: {self.real_path}")
            real_df = pd.read_csv(self.real_path)
            real_df['label'] = 1  # 1 for real
            print(f"Loaded {len(real_df)} real news articles")
            
            # Combine datasets
            self.df = pd.concat([fake_df, real_df], ignore_index=True)
            
            # Shuffle the dataset
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"\nCombined dataset: {len(self.df)} total articles")
            print(f"Columns: {self.df.columns.tolist()}")
            
            # Display class distribution
            print(f"\nClass distribution:")
            print(self.df['label'].value_counts())
            
            # Check for missing values
            print(f"\nMissing values:")
            print(self.df.isnull().sum())
            
            return True
            
        except Exception as e:
            print(f"Error loading ISOT data: {e}")
            print("Make sure Fake.csv and True.csv are in the ISOT folder")
            return False
    
    def debug_data_issues(self):
        """Debug common data issues"""
        print("\n" + "="*50)
        print("DEBUGGING DATA ISSUES")
        print("="*50)
        
        text_col = self.df.columns[0]
        label_col = self.df.columns[-1]
        
        # Check text lengths
        text_lengths = self.df[text_col].astype(str).str.len()
        print(f"\nText length statistics:")
        print(f"  Mean: {text_lengths.mean():.2f}")
        print(f"  Min: {text_lengths.min()}")
        print(f"  Max: {text_lengths.max()}")
        print(f"  Texts with < 10 characters: {(text_lengths < 10).sum()}")
        
        # Check unique labels
        print(f"\nUnique labels: {self.df[label_col].unique()}")
        print(f"Label types: {self.df[label_col].dtype}")
        
        # Show sample texts for each class
        print(f"\nSample texts by class:")
        for label in self.df[label_col].unique():
            sample_texts = self.df[self.df[label_col] == label][text_col].head(2)
            print(f"\nClass '{label}':")
            for i, text in enumerate(sample_texts):
                print(f"  {i+1}. {str(text)[:100]}...")
    
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
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\nPreparing data...")
        
        # For ISOT dataset, use 'text' column for content and 'label' for labels
        text_col = 'text'  # ISOT uses 'text' column
        label_col = 'label'  # We created this column
        
        print(f"Using '{text_col}' as text column")
        print(f"Using '{label_col}' as label column")
        
        # Debug data issues first
        self.debug_data_issues()
        
        # Preprocess text
        print("\nPreprocessing text...")
        self.df['clean_text'] = self.df[text_col].apply(self.preprocess_text)
        
        # Show sample preprocessed text
        print("\nSample preprocessed texts:")
        for i in range(min(3, len(self.df))):
            print(f"Original: {str(self.df[text_col].iloc[i])[:100]}...")
            print(f"Cleaned:  {self.df['clean_text'].iloc[i][:100]}...")
            print("-" * 50)
        
        # Remove empty texts
        before_clean = len(self.df)
        self.df = self.df[self.df['clean_text'].str.len() > 5]  # At least 5 characters
        after_clean = len(self.df)
        print(f"Records before cleaning: {before_clean}")
        print(f"Records after cleaning: {after_clean}")
        print(f"Removed {before_clean - after_clean} short/empty texts")
        
        # Labels are already properly set (0 for fake, 1 for real)
        y = self.df[label_col]
        
        print(f"\nFinal class distribution:")
        print(f"  Fake (0): {(y == 0).sum()}")
        print(f"  Real (1): {(y == 1).sum()}")
        
        # ISOT dataset is already balanced, but check anyway
        fake_count = (y == 0).sum()
        real_count = (y == 1).sum()
        ratio = max(fake_count, real_count) / min(fake_count, real_count)
        
        if ratio > 2:
            print(f"⚠️  Class imbalance detected (ratio: {ratio:.2f})")
        else:
            print("✅ Classes are well balanced!")
        
        # Prepare features and labels
        X = self.df['clean_text']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nFinal dataset split:")
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Training fake/real ratio: {(self.y_train == 0).sum()}/{(self.y_train == 1).sum()}")
        
        return True
    
    def balance_classes(self, df, y):
        """Balance classes using resampling"""
        df_temp = df.copy()
        df_temp['label'] = y
        
        # Separate classes
        df_fake = df_temp[df_temp['label'] == 0]
        df_real = df_temp[df_temp['label'] == 1]
        
        # Resample to balance
        if len(df_fake) > len(df_real):
            df_real_resampled = resample(df_real, n_samples=len(df_fake), random_state=42)
            df_balanced = pd.concat([df_fake, df_real_resampled])
        else:
            df_fake_resampled = resample(df_fake, n_samples=len(df_real), random_state=42)
            df_balanced = pd.concat([df_fake_resampled, df_real])
        
        return df_balanced.drop('label', axis=1), df_balanced['label']
    
    def vectorize_text(self, max_features=10000):
        """Convert text to TF-IDF features"""
        print("\nVectorizing text...")
        
        # Initialize TF-IDF vectorizer with better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,  # Word must appear in at least 2 documents
            max_df=0.95,  # Ignore words in more than 95% of documents
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic words with 2+ chars
        )
        
        # Fit and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Show most important features
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"Sample features: {feature_names[:10]}")
        
        return X_train_tfidf, X_test_tfidf
    
    def train_model(self, model_type='logistic'):
        """Train the classification model"""
        print(f"\nTraining {model_type} model...")
        
        # Vectorize text
        X_train_tfidf, X_test_tfidf = self.vectorize_text()
        
        # Choose model with better parameters
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=2000, 
                random_state=42,
                C=1.0,  # Regularization strength
                class_weight='balanced'  # Handle class imbalance
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200, 
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced'
            )
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")
        
        # Train model
        self.model.fit(X_train_tfidf, self.y_train)
        
        # Evaluate on training data
        train_pred = self.model.predict(X_train_tfidf)
        train_acc = accuracy_score(self.y_train, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        
        # Evaluate on test data
        test_pred = self.model.predict(X_test_tfidf)
        test_acc = accuracy_score(self.y_test, test_pred)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Check for overfitting
        if train_acc - test_acc > 0.1:
            print("⚠️  WARNING: Possible overfitting detected!")
            print(f"Training accuracy ({train_acc:.4f}) much higher than test accuracy ({test_acc:.4f})")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, test_pred, target_names=['FAKE', 'REAL']))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, test_pred)
        print(cm)
        print("\n[Rows: Actual, Columns: Predicted]")
        print("[0,0]: True Fake  [0,1]: False Real")
        print("[1,0]: False Fake [1,1]: True Real")
        
        return test_acc
    
    def predict(self, text):
        """Predict if a news article is fake or real"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet! Call train_model() first.")
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        if len(clean_text.strip()) == 0:
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0,
                'probabilities': {'fake': 50, 'real': 50},
                'error': 'Text too short after preprocessing'
            }
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([clean_text])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        return {
            'prediction': 'FAKE' if prediction == 0 else 'REAL',
            'confidence': max(probability) * 100,
            'probabilities': {
                'fake': probability[0] * 100,
                'real': probability[1] * 100
            },
            'preprocessed_text': clean_text[:100] + '...' if len(clean_text) > 100 else clean_text
        }
    
    def test_samples(self):
        """Test model with predefined samples"""
        print("\n" + "="*50)
        print("TESTING WITH SAMPLE NEWS")
        print("="*50)
        
        test_samples = [
            ("Breaking: Scientists discover aliens living in White House basement", "Expected: FAKE"),
            ("Apple announces new iPhone with improved camera technology", "Expected: REAL"),
            ("NASA successfully lands rover on Mars surface", "Expected: REAL"),
            ("Drinking coffee cures all diseases according to new study", "Expected: FAKE"),
            ("Local man wins lottery for third time this month", "Expected: FAKE")
        ]
        
        for text, expected in test_samples:
            result = self.predict(text)
            print(f"\nText: {text}")
            print(f"{expected}")
            print(f"Predicted: {result['prediction']} ({result['confidence']:.1f}% confidence)")
            print("-" * 70)
    
    def save_model(self, model_path='fake_news_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Save trained model and vectorizer"""
        print("\nSaving model...")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='fake_news_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Load trained model and vectorizer"""
        print("\nLoading model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print("Model loaded successfully!")
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("\n" + "="*50)
        print("FAKE NEWS DETECTOR - Interactive Mode")
        print("="*50)
        print("\nEnter news text to check (or 'quit' to exit)")
        print("Type 'test' to run predefined samples")
        
        while True:
            print("\n" + "-"*50)
            text = input("\nEnter news text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if text.lower() == 'test':
                self.test_samples()
                continue
            
            if not text:
                print("Please enter some text!")
                continue
            
            try:
                result = self.predict(text)
                print("\n" + "="*50)
                print(f"PREDICTION: {result['prediction']}")
                print(f"CONFIDENCE: {result['confidence']:.2f}%")
                print(f"\nProbabilities:")
                print(f"  Fake: {result['probabilities']['fake']:.2f}%")
                print(f"  Real: {result['probabilities']['real']:.2f}%")
                if 'preprocessed_text' in result:
                    print(f"\nPreprocessed: {result['preprocessed_text']}")
                print("="*50)
            except Exception as e:
                print(f"Error making prediction: {e}")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("FAKE NEWS DETECTION SYSTEM - ISOT DATASET")
    print("="*60)
    
    # Initialize detector for ISOT dataset
    detector = FakeNewsDetector(fake_path='ISOT/Fake.csv', real_path='ISOT/True.csv')
    
    # Load data
    if not detector.load_data():
        print("Failed to load data. Exiting...")
        exit(1)
    
    # Prepare data
    if not detector.prepare_data():
        print("Failed to prepare data. Exiting...")
        exit(1)
    
    # Train model
    print("\nChoose model type:")
    print("1. Logistic Regression (faster, good accuracy)")
    print("2. Random Forest (slower, potentially better accuracy)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    model_type = 'random_forest' if choice == '2' else 'logistic'
    
    detector.train_model(model_type=model_type)
    
    # Test with samples
    test_choice = input("\nTest with predefined samples? (y/n, default=y): ").strip().lower()
    if test_choice != 'n':
        detector.test_samples()
    
    # Save model
    save_choice = input("\nSave model? (y/n, default=y): ").strip().lower()
    if save_choice != 'n':
        detector.save_model()
    
    # Interactive mode
    interactive_choice = input("\nStart interactive prediction mode? (y/n, default=y): ").strip().lower()
    if interactive_choice != 'n':
        detector.interactive_mode()
    
    print("\nThank you for using the Fake News Detection System!")