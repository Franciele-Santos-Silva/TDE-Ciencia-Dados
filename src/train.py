from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

from data_loader import load_dataset
from preprocess import preprocess_dataframe
from utils import ensure_dirs


def main():
    print("Carregando dataset...")

    ensure_dirs()

    df = load_dataset()
    df = preprocess_dataframe(df)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["clean_text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    print("Tokenizando...")

    MODEL_NAME = "prajjwal1/bert-tiny"

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=256
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=256
    )

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels
    })

    test_dataset = Dataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": test_labels
    })

    print("Carregando BERT...")

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="outputs",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=500,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    print("Treinando...")

    trainer.train()

    print("Salvando modelo...")

    model.save_pretrained(
        "models/bert"
    )

    tokenizer.save_pretrained(
        "models/bert"
    )

    print("Treinamento concluído!")


if __name__ == "__main__":
    main()