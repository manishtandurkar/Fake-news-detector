import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  ShieldCheck, 
  ShieldAlert,  // Changed from ShieldX
  Search, 
  AlertCircle, 
  CheckCircle2, 
  XCircle,
  Loader2,
  Newspaper,
  TrendingUp,
  Eye
} from 'lucide-react';
import './App.css';

const API_BASE_URL = 'https://fake-news-detector-3llp.onrender.com';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Function to predict using the trained ML model via API
  const predictNews = async (text) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        return data.result;
      } else {
        throw new Error(data.error || 'Prediction failed');
      }
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    if (text.length < 10) {
      setError('Text too short. Please provide at least 10 characters.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const result = await predictNews(text);
      setResult(result);
    } catch (err) {
      console.error('Prediction Error:', err);
      if (err.message.includes('Failed to fetch') || err.message.includes('ECONNREFUSED')) {
        setError('Cannot connect to the AI model server. Please make sure the Flask API is running on localhost:5000. Run "python app.py" to start the server.');
      } else {
        setError(err.message || 'Failed to analyze text. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    setText('');
    setResult(null);
    setError('');
  };

  const getResultIcon = () => {
    if (!result) return null;
    return result.prediction === 'REAL' ? 
      <ShieldCheck className="result-icon real" /> : 
      <ShieldAlert className="result-icon fake" />;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return '#10b981'; // Green
    if (confidence >= 60) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
  };

  const exampleTexts = [
    "NASA announces successful landing of Perseverance rover on Mars, collecting soil samples for future analysis according to official mission reports.",
    "Breaking: Scientists discover aliens living in the White House basement, government confirms existence in shocking revelation that will change everything!",
    "Apple unveils new iPhone with revolutionary camera technology and improved battery life lasting 2 days, according to company officials and early reviews.",
    "You won't believe this amazing discovery! Doctors hate this one simple trick that cures all diseases instantly - leaked government documents reveal the hidden truth!",
    "According to a peer-reviewed study published by Stanford University researchers, new renewable energy storage methods show promising results in laboratory testing."
  ];

  return (
    <div className="app">
      {/* Header */}
      <motion.header 
        className="header"
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, type: "spring", stiffness: 100 }}
      >
        <div className="container">
          <div className="header-content">
            <motion.div 
              className="logo"
              whileHover={{ scale: 1.05 }}
              transition={{ duration: 0.2 }}
            >
              <Shield className="logo-icon" />
              <span className="logo-text">TruthGuard</span>
            </motion.div>
            <div className="header-subtitle">
              <Newspaper className="subtitle-icon" />
              ML-Powered Fake News Detection
            </div>
          </div>
        </div>
      </motion.header>

      {/* Hero Section */}
      <motion.section 
        className="hero"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 0.2 }}
      >
        <div className="container">
          <div className="hero-content">
            <motion.h1 
              className="hero-title"
              initial={{ y: 50, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.4, type: "spring", stiffness: 80 }}
            >
              Detect Fake News with AI
            </motion.h1>
            <motion.p 
              className="hero-description"
              initial={{ y: 50, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.6, type: "spring", stiffness: 80 }}
            >
              Our trained machine learning model analyzes news articles and headlines using advanced NLP techniques. 
              Powered by scikit-learn and TF-IDF vectorization on the ISOT dataset.
            </motion.p>
          </div>
        </div>
      </motion.section>

      {/* Main Content */}
      <main className="main">
        <div className="container">
          <motion.div 
            className="analyzer-card"
            initial={{ y: 100, opacity: 0, scale: 0.9 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.8, type: "spring", stiffness: 80 }}
          >
            <div className="card-header">
              <Eye className="card-icon" />
              <h2>News Authenticity Analyzer</h2>
            </div>

            <form onSubmit={handleSubmit} className="analyzer-form">
              <div className="input-group">
                <label htmlFor="newsText" className="input-label">
                  Enter news article or headline to analyze:
                </label>
                <textarea
                  id="newsText"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste your news article, headline, or any text you want to verify here..."
                  className="text-input"
                  rows={6}
                  maxLength={5000}
                />
                <div className="char-counter">
                  {text.length}/5000 characters
                </div>
              </div>

              <div className="button-group">
                <motion.button
                  type="submit"
                  disabled={loading || !text.trim()}
                  className="btn btn-primary"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {loading ? (
                    <>
                      <Loader2 className="btn-icon spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Search className="btn-icon" />
                      Analyze Text
                    </>
                  )}
                </motion.button>

                {(text || result) && (
                  <motion.button
                    type="button"
                    onClick={clearForm}
                    className="btn btn-secondary"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Clear
                  </motion.button>
                )}
              </div>
            </form>

            {/* Error Display */}
            <AnimatePresence>
              {error && (
                <motion.div 
                  className="error-message"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <AlertCircle className="error-icon" />
                  {error}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Results Display */}
            <AnimatePresence>
              {result && (
                <motion.div 
                  className="result-card"
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -50 }}
                  transition={{ duration: 0.5 }}
                >
                  <div className="result-header">
                    {getResultIcon()}
                    <h3 className={`result-title ${result.prediction.toLowerCase()}`}>
                      {result.prediction === 'REAL' ? 'Likely Real News' : 'Likely Fake News'}
                    </h3>
                  </div>

                  <div className="confidence-section">
                    <div className="confidence-label">
                      <TrendingUp className="confidence-icon" />
                      Confidence Level
                    </div>
                    <div className="confidence-bar">
                      <motion.div 
                        className="confidence-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence}%` }}
                        transition={{ duration: 1, delay: 0.2 }}
                        style={{ backgroundColor: getConfidenceColor(result.confidence) }}
                      />
                    </div>
                    <div className="confidence-text">
                      {result.confidence.toFixed(1)}%
                    </div>
                  </div>

                  <div className="probabilities">
                    <div className="probability-item">
                      <CheckCircle2 className="prob-icon real" />
                      <span>Real: {result.probabilities.real.toFixed(1)}%</span>
                    </div>
                    <div className="probability-item">
                      <XCircle className="prob-icon fake" />
                      <span>Fake: {result.probabilities.fake.toFixed(1)}%</span>
                    </div>
                  </div>

                  {result.preprocessed_text && (
                    <div className="preprocessed-section">
                      <h4>Processed Text Preview:</h4>
                      <div className="preprocessed-text">
                        {result.preprocessed_text}
                      </div>
                    </div>
                  )}

                  <div className="disclaimer-section">
                    <h4>ℹ️ AI Model Notice</h4>
                    <p>This prediction is powered by a machine learning model trained on the ISOT dataset using TF-IDF vectorization and Logistic Regression. Always verify news from multiple reliable sources.</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Example Texts */}
            {!result && !loading && (
              <div className="examples-section">
                <h3>Try These Examples:</h3>
                <div className="examples-grid">
                  {exampleTexts.map((example, index) => (
                    <motion.button
                      key={index}
                      className="example-card"
                      onClick={() => setText(example)}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.4, delay: index * 0.1 }}
                      whileHover={{ scale: 1.02, x: 5 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="example-text">
                        {example}
                      </div>
                    </motion.button>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>&copy; 2025 TruthGuard. ML-powered fact-checking with trained ISOT model.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;