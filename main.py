import re
import joblib

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from colorama import Fore, init

init(autoreset=True)
GREEN   = Fore.GREEN
RED     = Fore.RED
YELLOW  = Fore.YELLOW

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
    sentiment = f"{GREEN}Positive" if prediction == 1 else f"{RED}Negative"

    return sentiment, probability_percentage

# Main Code
while True:
    review = input("\nEnter a review          : ")
    
    if review != "":
        sentiment, probability_percentage   = predict_sentiment(review)
        negative_percentage                 = probability_percentage[0] * 100
        positive_percentage                 = probability_percentage[1] * 100

        print("SENTIMENT               :", sentiment, f"{YELLOW}({negative_percentage:.3g}% Negativity, {positive_percentage:.3g}% Positivity)")
        print("PROBABILITY_PERCENTAGE  :", probability_percentage)
    else:
        print("Please enter a review to predict.")