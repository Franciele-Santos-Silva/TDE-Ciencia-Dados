import pandas as pd


def load_dataset():
    fake = pd.read_csv("data/raw/Fake.csv")
    true = pd.read_csv("data/raw/True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], axis=0)

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return df.sample(
    n=5000,
    random_state=42
)