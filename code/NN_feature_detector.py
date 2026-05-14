import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LLMDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def train_and_evaluate(train_file, dev_file, output_dir="./model_output", model_name="hfl/chinese-roberta-wwm-ext"):
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    train_texts, train_labels = load_data(train_file)
    dev_texts, dev_labels     = load_data(dev_file)
    print(f"训练集样本数: {len(train_texts)}")
    print(f"开发集样本数: {len(dev_texts)}")

    train_dataset = LLMDetectionDataset(train_texts, train_labels, tokenizer)
    dev_dataset   = LLMDetectionDataset(dev_texts, dev_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,

        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,

        load_best_model_at_end=True,
        metric_for_best_model="f1",

        fp16=True,
        no_cuda=False
    )

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        return {
            "f1": f1_score(pred.label_ids, preds, average="macro")
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("开始训练模型...")
    trainer.train()

    print("在开发集上评估模型...")
    eval_results = trainer.evaluate()
    print(f"开发集评估结果: {eval_results}")

    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)            
    tokenizer.save_pretrained(final_dir)
    return model, tokenizer


def predict_on_test_data(test_file, model_dir, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []

    model.eval()
    with torch.no_grad():
        for item in test_data:
            text = item["text"]
            item_id = item.get("id", None)

            inputs = tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

            result_item = {"text": text, "label": prediction}
            if item_id is not None:
                result_item["id"] = item_id
            results.append(result_item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"预测结果已保存到 {output_file}")

#可视化
def visualize_results(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    report_dict = classification_report(all_labels, all_preds, output_dict=True, digits=4)
    report_df = pd.DataFrame(report_dict).transpose()
    print("Classification Report:\n", report_df)

    metrics = ["precision", "recall", "f1-score"]
    idx = report_df.index[:-3]
    x = np.arange(len(idx))
    width = 0.2

    plt.figure(figsize=(8, 4))
    for i, m in enumerate(metrics):
        plt.bar(x + i * width, report_df.loc[idx, m], width=width, label=m)
    plt.xticks(x + width, idx)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Class Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = [str(i) for i in report_df.index[:-3]]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    model, tokenizer = train_and_evaluate(
        train_file="data/train.json",
        dev_file="data/dev.json",
        output_dir="./model_output",
        model_name="hfl/chinese-roberta-wwm-ext"
    )
    predict_on_test_data(
        test_file="data/test.json",
        model_dir="./model_output/final_model",
        output_file="./results.json"
    )
    dev_texts, dev_labels = load_data("data/dev.json")
    dev_dataset = LLMDetectionDataset(dev_texts, dev_labels, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=64)
    visualize_results(model, dev_loader, device)
