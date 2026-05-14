import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, Trainer
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display


#定义结合特征的模型
class EnhancedLLMDetector(nn.Module):
    def __init__(self, pretrained_model_name, num_statistical_features=6):
        super(EnhancedLLMDetector, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        
        #特征提取后的维度
        hidden_size = self.bert.config.hidden_size
        
        #结合统计特征的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + num_statistical_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
    
    def forward(self, input_ids, attention_mask, statistical_features):
        #获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  #CLS token
        pooled_output = self.dropout(pooled_output)
        
        #结合统计特征
        combined_features = torch.cat([pooled_output, statistical_features], dim=1)
        
        #分类
        logits = self.classifier(combined_features)
        return logits

#定义数据集类
class EnhancedLLMDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, statistical_features, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.statistical_features = statistical_features
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        stat_features = self.statistical_features[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'statistical_features': torch.tensor(stat_features, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

#加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    return texts, labels

#计算文本的统计特征
def extract_text_features(texts):
    features = []
    
    for text in texts:
        #文本长度
        text_length = len(text)
        
        #平均句子长度
        sentences = text.split('。')
        avg_sentence_length = sum(len(s) for s in sentences) / max(1, len(sentences))
        
        #词汇多样性
        unique_chars = len(set(text))
        char_diversity = unique_chars / max(1, text_length)
        
        #标点符号比例
        punctuation = sum(1 for char in text if char in '，。！？；：""''（）【】《》')
        punc_ratio = punctuation / max(1, text_length)
        
        #数字比例
        digits = sum(1 for char in text if char.isdigit())
        digit_ratio = digits / max(1, text_length)
        
        #特殊符号比例
        special_chars = sum(1 for char in text if not char.isalnum() and char not in '，。！？；：""''（）【】《》 \t\n')
        special_ratio = special_chars / max(1, text_length)
        
        #重复词计数
        words = text.split()
        if len(words) > 1:
            word_repetition = sum(1 for i in range(len(words)-1) if words[i] == words[i+1]) / (len(words) - 1)
        else:
            word_repetition = 0
            
        #字符2-gram熵
        n = 2  
        if len(text) >= n:
            ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
            ngram_freq = {}
            for ngram in ngrams:
                ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1
            total_ngrams = len(ngrams)
            entropy = -sum((count/total_ngrams) * np.log2(count/total_ngrams) for count in ngram_freq.values())
            normalized_entropy = entropy / np.log2(min(total_ngrams, len(ngram_freq)))
        else:
            normalized_entropy = 0
        
        features.append([
            text_length, 
            avg_sentence_length, 
            char_diversity, 
            punc_ratio, 
            digit_ratio,
            special_ratio,
            word_repetition,
            normalized_entropy
        ])
    return np.array(features)

def train_model(model, train_dataloader, optimizer, scheduler, device, num_epochs=3):
    model.train()
    
    for epoch in range(num_epochs):
        print(f"开始训练 Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            statistical_features = batch['statistical_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                statistical_features=statistical_features
            )
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    return model

#评估
def evaluate_model(model, eval_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            statistical_features = batch['statistical_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                statistical_features=statistical_features
            )
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds, digits=4)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return report, macro_f1

def train_and_evaluate(train_file, dev_file, output_dir="./enhanced_model_output", model_name="hfl/chinese-roberta-wwm-ext",device=None):
    print(f"使用设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_texts, train_labels = load_data(train_file)
    dev_texts, dev_labels = load_data(dev_file)
    
    print(f"训练集样本数: {len(train_texts)}")
    print(f"开发集样本数: {len(dev_texts)}")
    
    print("提取训练集统计特征...")
    train_features = extract_text_features(train_texts)
    print("提取开发集统计特征...")
    dev_features = extract_text_features(dev_texts)
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    dev_features_scaled = scaler.transform(dev_features)
    
    import joblib
    joblib.dump(scaler, f"{output_dir}/feature_scaler.pkl")
    
    train_dataset = EnhancedLLMDetectionDataset(
        train_texts, train_labels, tokenizer, train_features_scaled
    )
    dev_dataset = EnhancedLLMDetectionDataset(
        dev_texts, dev_labels, tokenizer, dev_features_scaled
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True
    )
    dev_dataloader = DataLoader(
        dev_dataset, 
        batch_size=32, 
        shuffle=False
    )
    
    model = EnhancedLLMDetector(model_name, num_statistical_features=train_features.shape[1]).to(device)  

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * 3  
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    print("开始训练模型...")
    model = train_model(model, train_dataloader, optimizer, scheduler, device, num_epochs=3)
    
    print("在开发集上评估模型...")
    report, macro_f1 = evaluate_model(model, dev_dataloader, device)
    print(f"开发集分类报告:\n{report}")
    print(f"开发集宏平均F1分数: {macro_f1:.4f}")
    
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    tokenizer.save_pretrained(output_dir)
    print(f"模型已保存到 {output_dir}")
    
    return model, tokenizer, scaler

def predict_on_test_data(test_file, model_dir, output_file,device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    scaler = joblib.load(f"{model_dir}/feature_scaler.pkl")

    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_texts = [item['text'] for item in test_data]
    
    test_features = extract_text_features(test_texts)
    test_features_scaled = scaler.transform(test_features)
    
    model = EnhancedLLMDetector("hfl/chinese-roberta-wwm-ext", num_statistical_features=test_features.shape[1]).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
    model.eval()
    
    batch_size = 32
    results = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_texts), batch_size), desc="预测测试数据"):
            batch_texts = test_texts[i:i+batch_size]
            batch_features = test_features_scaled[i:i+batch_size]
            
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            stat_features = torch.tensor(batch_features, dtype=torch.float).to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                statistical_features=stat_features
            )
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for j, pred in enumerate(predictions):
                idx = i + j
                result_item = {
                    'text': test_data[idx]['text'],
                    'label': int(pred)
                }
                if 'id' in test_data[idx]:
                    result_item['id'] = test_data[idx]['id']
                
                results.append(result_item)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"预测结果已保存到 {output_file}")




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer, scaler = train_and_evaluate(
        train_file="data/train.json", 
        dev_file="data/dev.json",
        output_dir="./enhanced_model_output",
        model_name="hfl/chinese-roberta-wwm-ext",
        device=device,    
    )
    predict_on_test_data(
        test_file="data/test.json",
        model_dir="./enhanced_model_output",
        output_file="./enhanced_results.json",
        device=device,    
    )

    #准备可视化的数据
    dev_texts, dev_labels = load_data("data/dev.json")
    dev_features = extract_text_features(dev_texts)
    dev_features_scaled = scaler.transform(dev_features)
    feature_names = [
        "text_length", "avg_sentence_len", "char_diversity", "punc_ratio",
        "digit_ratio", "special_ratio", "word_repetition", "ngram_entropy"
    ]
    df_dev = pd.DataFrame(dev_features_scaled, columns=feature_names)
    df_dev["label"] = dev_labels

    #得到开发集预测
    report, macro_f1 = evaluate_model(model, DataLoader(
        EnhancedLLMDetectionDataset(dev_texts, dev_labels, tokenizer, dev_features_scaled),
        batch_size=32, shuffle=False
    ), device)
    print("开发集分类报告：\n", report)
    print(f"宏平均 F1 分数: {macro_f1:.4f}")

    all_logits = []
    for i, text in enumerate(dev_texts):
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].clone().detach().to(device)
        attention_mask = enc["attention_mask"].clone().detach().to(device)
        stat_feat = torch.tensor(dev_features_scaled[i], dtype=torch.float).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                statistical_features=stat_feat
            )
        all_logits.append(logits.cpu().numpy().squeeze())

    pred_labels = np.argmax(np.vstack(all_logits), axis=1)

    #classification_report
    cr_dict = classification_report(
        dev_labels,
        pred_labels,
        output_dict=True,
        digits=4
    )
    df_cr = pd.DataFrame(cr_dict).transpose()
    print("分类报告表格预览：")
    display(df_cr)

    #混淆矩阵热力图
    cm = confusion_matrix(dev_labels, pred_labels)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = [str(i) for i in sorted(set(dev_labels))]
    plt.xticks(range(len(ticks)), ticks)
    plt.yticks(range(len(ticks)), ticks)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                    ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()
