# Fake News Detector - Beautiful UI

A stunning React + Flask application for detecting fake news using AI/ML.

## 🚀 Features

- **Beautiful Modern UI** with animations and gradients
- **AI-Powered Detection** using machine learning
- **Real-time Analysis** with confidence scores
- **Responsive Design** works on all devices
- **Interactive Examples** to test the system

## 📁 Project Structure

```
fake-news-detector/
├── frontend/                 # React UI
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Beautiful styling
│   │   └── index.js         # Entry point
│   ├── public/
│   │   └── index.html       # HTML template
│   └── package.json         # Dependencies
├── backend/
│   ├── app.py              # Flask API server
│   ├── main.py             # ML model training
│   └── requirements.txt    # Python dependencies
├── ISOT/                   # Dataset folder
│   ├── Fake.csv           # Fake news dataset
│   └── True.csv           # Real news dataset
└── README.md              # This file
```

## 🛠️ Setup Instructions

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   cd "c:\Projects\Extra\Fake news detector"
   pip install -r requirements.txt
   ```

2. **Train the model first:**
   ```bash
   python main.py
   # Choose option 2 for ISOT dataset
   # Train with Logistic Regression (option 1)
   # Save the model when prompted
   ```

3. **Start the Flask API server:**
   ```bash
   python app.py
   ```
   The API will be available at: http://localhost:5000

### Frontend Setup

1. **Navigate to frontend folder:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```
   The UI will open at: http://localhost:3000

## 🌟 How to Use

1. **Start both servers** (Flask API + React frontend)
2. **Open your browser** to http://localhost:3000
3. **Enter news text** in the textarea
4. **Click "Analyze Text"** to get results
5. **View confidence scores** and authenticity prediction

## 🎨 UI Features

- **Gradient Background** with glass morphism effects
- **Smooth Animations** using Framer Motion
- **Real-time Confidence Bars** with color coding
- **Interactive Example Cards** for quick testing
- **Responsive Design** for mobile and desktop
- **Beautiful Icons** from Lucide React

## 🔧 Troubleshooting

**"Cannot connect to API server" error:**
- Make sure Flask server is running on localhost:5000
- Check if `fake_news_model.pkl` and `vectorizer.pkl` exist
- Train the model first using `python main.py`

**Frontend not loading:**
- Make sure you're in the `frontend` folder
- Run `npm install` to install dependencies
- Check if Node.js is installed

## 🤖 API Endpoints

- `GET /` - Health check
- `POST /api/predict` - Analyze text for fake news
- `GET /api/health` - Check model status
- `POST /api/train` - Retrain the model

## 🎯 Example Usage

```json
POST /api/predict
{
  "text": "NASA announces successful Mars landing"
}

Response:
{
  "success": true,
  "result": {
    "prediction": "REAL",
    "confidence": 87.5,
    "probabilities": {
      "fake": 12.5,
      "real": 87.5
    }
  }
}
```

## 📊 Model Information

- **Algorithm:** Logistic Regression with TF-IDF
- **Dataset:** ISOT Fake News Dataset (~44k articles)
- **Accuracy:** 85-95% on test data
- **Features:** Text preprocessing, vectorization, classification

Enjoy using your beautiful fake news detector! 🛡️✨