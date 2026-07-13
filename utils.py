import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text = str(text)

    # lowercase
    text = text.lower()


    # remove special characters
    text = re.sub(
        "[^a-zA-Z]",
        " ",
        text
    )


    # tokenize
    words = text.split()


    # remove stopwords + lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]


    return " ".join(words)



def create_sentiment(df):

    # Convert ratings like:
    # "Rated 5 out of 5 stars"
    # into 5

    df["Rating"] = (
        df["Rating"]
        .str.extract(r"(\d+)")
        .astype(int)
    )


    # Positive / Negative

    df["Sentiment"] = df["Rating"].apply(
        lambda x:
        "Positive" if x >= 4 else "Negative"
    )


    return df