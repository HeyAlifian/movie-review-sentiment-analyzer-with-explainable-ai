# from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from lime.lime_text import LimeTextExplainer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import joblib
import pandas
import re
import nltk

nltk.download("stopwords")

class RAWDATAProcessor:
    def __init__(self, dataset_path: str, drop_na: bool = True, separator: any = ",") -> None:

        assert isinstance(dataset_path, str), f"Invalid type of 'dataset_path', expected str. Found {type(dataset_path)}"
        assert isinstance(drop_na, bool), f"Invalid type of 'drop_na', expected bool. Found {type(dataset_path)}"

        self.data           = pandas.read_csv(dataset_path, sep=separator)
        self.isdrop_na      = drop_na
        self.stop_words     = set(stopwords.words('english'))
        self.tokenizer      = RegexpTokenizer(r"[A-Za-z']+")
        self.data[
            "clean_review"
            ]               = self.data["review"].apply(self.CleanData)
        self.Head()
        self.Return()
    
    def CleanData(self, data):

        data                = re.sub(r'<.*?>', ' ', data)
        tokens              = self.tokenizer.tokenize(data.lower())
        tokens              = [t for t in tokens if t not in self.stop_words]
        return " ".join(tokens)

    def Head(self):
        print(self.data.head())

    def Return(self):
        return self.data
    
def SaveModel(model, file_name: str, path: str = r"Models") -> None:
    if model != "":
        try:
            joblib.dump(model, f"{path}/{file_name}")
            print(f"SUCCESSFULLY SAVED DATA TO: {path}/{file_name}")
        except Exception as err:
            print(f"[ERROR]:", err)
    else:
        print(f"NO MODELS INPUTTED.")


# TRAIN PROCESS
dataset_path    = r"Datasets\imdb-datasets.csv"

dataset         = RAWDATAProcessor(dataset_path=dataset_path, drop_na=True, separator=',').Return()

X               = dataset["clean_review"]
y               = dataset["sentiment"].map({"negative": 0, "positive": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)

vectorizer      = TfidfVectorizer(max_features = 10_000, ngram_range=(1,2))
X_train_tfidf   = vectorizer.fit_transform(X_train)
X_test_tfidf    = vectorizer.transform(X_test)

model           = LogisticRegression(max_iter = 200)
model.fit(X_train_tfidf, y_train)

y_prediction    = model.predict(X_test_tfidf)

print("ACCURACY:", accuracy_score(y_test, y_prediction))
print("\nCLASSIFICATION REPORT:\n-------------------------------------------------\n",classification_report(y_test, y_prediction))

X_all           = dataset['clean_review']
y_all           = dataset["sentiment"].map({'negative':0, 'positive':1})

# FULL DATA TRAINING
print("FULL DATA TRAINING...")
X_all_tfidf     = vectorizer.fit_transform(X_all)
final_model     = LogisticRegression(max_iter=200)
final_model.fit(X_all_tfidf, y_all)

SaveModel(final_model, "sentiment_model.pkl")
SaveModel(vectorizer, "tfidf_vectorizer.pkl")