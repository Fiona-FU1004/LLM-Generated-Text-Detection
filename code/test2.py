import json
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix
)

from enhanced_detector import (
    EnhancedLLMDetector,
    EnhancedLLMDetectionDataset,
    load_data,
    extract_text_features,
    evaluate_model
)

def visualize_results(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stat_features = batch["statistical_features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                statistical_features=stat_features
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    #分类报告
    report_dict = classification_report(all_labels, all_preds, output_dict=True, digits=4)
    report_df = pd.DataFrame(report_dict).transpose()
    print("=== 分类报告 ===\n", report_df)

    #Precision / Recall / F1 柱状图
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

    #混淆矩阵热力图
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def full_evaluation(model, tokenizer, scaler, texts, labels, batch_size=16, device="cpu"):
    features = extract_text_features(texts)
    features_scaled = scaler.transform(features)

    dataset = EnhancedLLMDetectionDataset(texts, labels, tokenizer, features_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    report, macro_f1 = evaluate_model(model, dataloader, device)
    print("\n———————— 全量测试集评估结果 ——————————")
    print(report)
    print(f"Macro F1: {macro_f1:.4f}\n")

    visualize_results(model, dataloader, device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_dir = "./enhanced_model_output"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    scaler = joblib.load(f"{model_dir}/feature_scaler.pkl")

    model = EnhancedLLMDetector(
        pretrained_model_name="hfl/chinese-roberta-wwm-ext",
        num_statistical_features=scaler.mean_.shape[0]
    ).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
    model.eval()

    texts, labels = load_data("data/test_with_label.json")

    full_evaluation(
        model=model,
        tokenizer=tokenizer,
        scaler=scaler,
        texts=texts,
        labels=labels,
        batch_size=16,
        device=device
    )
