# 🧠 RNN-Based Sentiment Analysis Web Application

A machine learning web application that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative** using a **Recurrent Neural Network (RNN)** built with **PyTorch**. The application provides **real-time predictions** along with a **confidence score** through a simple and interactive web interface.

---

## 🚀 Live Demo

https://rnn-sentiment-analysis-l14u.onrender.com/

---

## ✨ Features

- Predicts **Positive** or **Negative** sentiment
- Displays **prediction confidence score**
- Real-time sentiment analysis
- Text preprocessing using **Porter Stemming**
- TF-IDF feature extraction
- RNN model implemented using **PyTorch**
- Flask-based REST backend
- Interactive frontend built with **HTML, CSS, and JavaScript**
- Fast and lightweight prediction pipeline

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- Flask

### Machine Learning
- PyTorch
- Recurrent Neural Network (RNN)
- TF-IDF Vectorizer
- NLTK
- Scikit-learn

---

## 📂 Project Structure

```text
RNN-Sentiment-Analysis/
│
├── app.py
├── sentiment_model.pth
├── tfidf.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

## ⚙️ How It Works

1. User enters a movie review.
2. The review is preprocessed using stemming.
3. TF-IDF converts the text into numerical feature vectors.
4. The trained RNN model predicts the sentiment.
5. The application returns:
   - Sentiment (Positive / Negative)
   - Confidence Score

---

## 📊 Model Pipeline

```text
User Review
      │
      ▼
Text Preprocessing
      │
      ▼
Porter Stemming
      │
      ▼
TF-IDF Vectorization
      │
      ▼
PyTorch RNN Model
      │
      ▼
Sentiment Prediction
      │
      ▼
Confidence Score
```

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/your-username/RNN-Sentiment-Analysis.git
```

Navigate to the project directory

```bash
cd RNN-Sentiment-Analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```text
http://127.0.0.1:5000
```

---

## 📈 Future Enhancements

- Support for multiple sentiment classes
- Advanced text preprocessing
- Attention-based RNN/LSTM model
- User authentication
- Prediction history
- Model comparison dashboard
- Docker containerization
- CI/CD deployment pipeline

---

## 👨‍💻 Author

**Saurav Jha**

B.Tech Student | Aspiring Full Stack & AI/ML Developer

- GitHub: https://github.com/SauravCoder69
- LinkedIn: https://www.linkedin.com/in/sauravjha03?utm_source=share_via&utm_content=profile&utm_medium=member_android

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.
