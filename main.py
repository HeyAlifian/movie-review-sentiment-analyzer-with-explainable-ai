import re
import joblib
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Load 'em models.
model      = joblib.load(r"Models\sentiment_model.pkl")
vectorizer = joblib.load(r"Models\tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
tokenizer = RegexpTokenizer(r"[A-Za-z']+")

def clean_text(text):

    text = re.sub(r'<.*?>', ' ', text)
    
    tokens = tokenizer.tokenize(text.lower())
    
    tokens = [t for t in tokens if t not in stop_words]

    return " ".join(tokens)

def predict_sentiment(review):

    cleaned = clean_text(review)
    
    X_vectorized = vectorizer.transform([cleaned])
    
    prediction = model.predict(X_vectorized)[0]
    probability_percentage = model.predict_proba(X_vectorized)[0]
    
    sentiment = "Positive :)" if prediction == 1 else "Negative :("

    return sentiment, probability_percentage

# Main Code
while True:
    review = input("Enter a review: ")
    
    if review != "":
        sentiment, probability_percentage = predict_sentiment(review)

        print("Sentiment:", sentiment)
        print("Percentage:", probability_percentage)
    else:
        print("Please enter a review to predict.")