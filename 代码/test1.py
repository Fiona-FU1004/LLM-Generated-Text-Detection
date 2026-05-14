import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  
import numpy as np
import json

from llm_detector import LLMDetectionDataset, load_data, evaluate_model  

def visualize_report_and_matrix(labels, preds, class_names=None):
    #1. 混淆矩阵
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    #2. 分类报告转 DataFrame
    report = classification_report(labels, preds, output_dict=True, digits=4)
    class_keys = [k for k in report.keys() if k.isdigit() or k in map(str, range(len(report)-3))]
    precisions = [report[k]['precision'] for k in class_keys]
    recalls    = [report[k]['recall']    for k in class_keys]
    f1s        = [report[k]['f1-score']  for k in class_keys]
    x = np.arange(len(class_keys))
    width = 0.25

    plt.figure(figsize=(8,4))
    plt.bar(x - width, precisions, width=width, label='Precision')
    plt.bar(x,        recalls,    width=width, label='Recall')
    plt.bar(x + width, f1s,       width=width, label='F1-score')
    plt.xticks(x, class_names or class_keys)
    plt.ylim(0,1.05)
    plt.ylabel("Score")
    plt.title("Per-Class Precision / Recall / F1")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "./model_output/final_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    texts, labels = load_data("data/test_with_label.json")

    dataset = LLMDetectionDataset(texts, labels, tokenizer)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    report, macro_f1 = evaluate_model(model, loader, device)
    print("=== 全测试集评估结果 ===")
    print(report)
    print(f"Macro F1: {macro_f1:.4f}")

    #可视化
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_b = batch['labels'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds_b = torch.argmax(logits, dim=1)
            all_preds.extend(preds_b.cpu().numpy())
            all_labels.extend(labels_b.cpu().numpy())

    class_names = ['Human', 'AI']
    visualize_report_and_matrix(all_labels, all_preds, class_names)
