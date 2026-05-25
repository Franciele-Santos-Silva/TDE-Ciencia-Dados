import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer
)

from datasets import Dataset

from data_loader import load_dataset
from preprocess import preprocess_dataframe


def main():
    print("Carregando modelo...")

    df = load_dataset()

    df = preprocess_dataframe(df)

    texts = df["clean_text"].tolist()
    labels = df["label"].tolist()

    tokenizer = BertTokenizer.from_pretrained(
        "models/bert"
    )

    model = BertForSequenceClassification.from_pretrained(
        "models/bert"
    )

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256
    )

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    })

    trainer = Trainer(model=model)

    predictions = trainer.predict(
        dataset
    )

    preds = np.argmax(
        predictions.predictions,
        axis=1
    )

    acc = accuracy_score(
        labels,
        preds
    )

    print("Accuracy:", acc)

    report = classification_report(
        labels,
        preds
    )

    print(report)

    with open(
        "metrics/classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    pd.DataFrame({
        "text": texts,
        "prediction": preds
    }).to_csv(
        "predictions/test_predictions.csv",
        index=False
    )

    cm = confusion_matrix(
        labels,
        preds
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        "Matriz de Confusão"
    )

    plt.savefig(
        "outputs/matriz_confusao.png"
    )

    plt.close()

    print("Avaliação concluída!")


if __name__ == "__main__":
    main()