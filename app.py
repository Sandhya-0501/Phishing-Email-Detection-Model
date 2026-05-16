from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)

# 🔹 Sample training data (you can expand this)
emails = [
    "Click here to verify your bank account",
    "Urgent! Update your password now",
    "Win a free lottery prize now",
    "Meeting scheduled tomorrow",
    "Project discussion at 10 AM",
    "Lunch plans with team",
]

labels = ["Phishing", "Phishing", "Phishing", "Safe", "Safe", "Safe"]

# 🔹 Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]
    accuracy = model.score(X, labels)

    return jsonify({
        "prediction": prediction,
        "accuracy": round(accuracy * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)