from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import pickle
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")

app = Flask(__name__)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out




with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

input_size = len(tfidf.get_feature_names_out())


model = RNN(input_size)

model.load_state_dict(
    torch.load("sentiment_model.pth", map_location=torch.device("cpu"))
)

model.eval()



ps = PorterStemmer()

def stemming(text):

    tokens = word_tokenize(text)

    words = []

    for word in tokens:
        words.append(ps.stem(word))

    return " ".join(words)




@app.route("/")
def home():
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    review = review.lower()

    review = stemming(review)

    review = tfidf.transform([review])

    review = torch.tensor(
        review.toarray(),
        dtype=torch.float32
    )

    review = review.unsqueeze(1)

    with torch.no_grad():

        output = model(review)

        probability = torch.sigmoid(output)

        prediction = (probability > 0.5).float()

    if prediction.item() == 1:
        sentiment = "Positive "
    else:
        sentiment = "Negative "

    confidence = round(probability.item() * 100, 2)

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence
    })


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)