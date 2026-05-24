import re
import string


def clean_text(text):
    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def preprocess_dataframe(df):
    df["clean_text"] = df["text"].apply(
        clean_text
    )

    df.to_csv(
        "data/processed/cleaned_news.csv",
        index=False
    )

    return df