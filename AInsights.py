from lime.lime_text import LimeTextExplainer
import joblib

# [i] Load the models again.
model      = joblib.load(r"Models\sentiment_model.pkl")
vectorizer = joblib.load(r"Models\tfidf_vectorizer.pkl")

# [i] Define class
class AInsights:
    """
    It provides you insights from a sample of text OR sentence that you inputted into the AInsights by giving you the values of how many percentage of a word for you to analyze and visualize.

    # Parameters
        - text (str)            = A sample of text OR sentence must be in a string format, <b>not in integer, list, or even dictionary.</b>
        - insights_model (str)  = A type of insights model you want to using LimeTextExplainer.
    # Result
    ```bash

    ```
    """

    def __init__(self, text) -> None:
        self.text_data = text
        self.explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

    def Explain(self) -> None:
        exp = self.explainer.explain_instance(
            self.text_data,
            lambda x: model.predict_proba(vectorizer.transform(x)),
            num_features=6
            )
        
        # exp.show_in_notebook(text=self.text_data)
        exp_as_list     = exp.as_list()
        max_length      = 25
        # max_length      = max(len(word) for word, _ in exp_as_list)

        print("\nAINSIGHTS REPORT\n--------------------------------------")
        for word, weight in exp_as_list:
            print(f"{word:<{max_length}}: {weight:>7.3g}")

# [i] Main Code
if __name__ == "__main__":
    text        = "This movie is brilliant, amazing, and heartwarming!"
    explainer   = AInsights(text=text)
    explainer.Explain()