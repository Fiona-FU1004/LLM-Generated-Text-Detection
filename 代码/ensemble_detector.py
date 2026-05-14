import json
import numpy as np
import torch
import joblib
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from enhanced_detector import (
    EnhancedLLMDetector,
    extract_text_features,
    EnhancedLLMDetectionDataset,
    load_data
)
from IPython.display import display




#集成模型类
class EnsembleDetector:
    def __init__(self, models_config):
        self.models = []
        self.tokenizers = []
        self.scalers = []
        self.model_types = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for config in models_config:
            model_type = config['type']
            model_path = config['path']
            abs_path   = os.path.abspath(model_path)
            safe_path = abs_path.replace("\\", "/")
            
            self.model_types.append(model_type)

            if model_type == 'enhanced':
                #加载增强型模型
                assert os.path.exists(abs_path), f"模型路径不存在: {abs_path}"
                tokenizer = AutoTokenizer.from_pretrained(safe_path, local_files_only=True)
                self.tokenizers.append(tokenizer)
                
                scaler = joblib.load(f"{abs_path}/feature_scaler.pkl")
                self.scalers.append(scaler)
                model = EnhancedLLMDetector("hfl/chinese-roberta-wwm-ext", num_statistical_features=8)
                model.load_state_dict(torch.load(os.path.join(abs_path, "model.pt"), map_location=self.device))                
                model.to(self.device)
                model.eval()
                self.models.append(model)
                
            elif model_type == 'transformer':
                #加载普通Transformer模型
                tokenizer = AutoTokenizer.from_pretrained(safe_path, local_files_only=True)
                self.tokenizers.append(tokenizer)
                self.scalers.append(None)  
                
                model = AutoModelForSequenceClassification.from_pretrained(safe_path,local_files_only=True)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                
            elif model_type == 'random_forest':
                #加载RandomForest模型
                self.tokenizers.append(None) 
                scaler = joblib.load(f"{abs_path}/feature_scaler.pkl")
                self.scalers.append(scaler)
                model = joblib.load(os.path.join(abs_path, "rf_model.pkl"))
                self.models.append(model)
    
            elif model_type == 'gradient_boosting':
                #加载GradientBoosting模型
                self.tokenizers.append(None)  
                scaler = joblib.load(f"{abs_path}/feature_scaler.pkl")
                self.scalers.append(scaler)
                model = joblib.load(os.path.join(abs_path, "gb_model.pkl"))
                self.models.append(model)

    
    def predict(self, texts, batch_size=32, weights=None):
        all_predictions = []
        
        for i, (model, tokenizer, scaler, model_type) in enumerate(zip(self.models, self.tokenizers, self.scalers, self.model_types)):
            print(f"使用模型 {i+1}/{len(self.models)} 进行预测...")
            
            if model_type == 'enhanced':
                #使用增强型模型预测
                predictions = self._predict_with_enhanced_model(texts, model, tokenizer, scaler, batch_size)
                
            elif model_type == 'transformer':
                #使用普通Transformer模型预测
                predictions = self._predict_with_transformer_model(texts, model, tokenizer, batch_size)
                
            elif model_type in ['random_forest', 'gradient_boosting']:
                #使用sklearn模型预测
                predictions = self._predict_with_sklearn_model(texts, model, scaler)
            
            all_predictions.append(predictions)
        
        #合并预测结果
        all_predictions = np.array(all_predictions)
        
        #如果没有提供权重，就平均加权
        if weights is None:
            weights = np.ones(len(self.models)) / len(self.models)
        
        #加权投票
        weighted_predictions = np.zeros(len(texts))
        for i, pred in enumerate(all_predictions):
            weighted_predictions += weights[i] * pred
        
        final_predictions = np.round(weighted_predictions).astype(int)
        
        return final_predictions
    
    def _predict_with_enhanced_model(self, texts, model, tokenizer, scaler, batch_size):
        """使用增强型模型进行预测"""
        predictions = []
        
        features = extract_text_features(texts)
        features_scaled = scaler.transform(features)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="预测中"):
                batch_texts = texts[i:i+batch_size]
                batch_features = features_scaled[i:i+batch_size]
                
                encodings = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                stat_features = torch.tensor(batch_features, dtype=torch.float).to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    statistical_features=stat_features
                )
                
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    def _predict_with_transformer_model(self, texts, model, tokenizer, batch_size):
        """使用普通Transformer模型进行预测"""
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="预测中"):
                batch_texts = texts[i:i+batch_size]
                
                encodings = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    def _predict_with_sklearn_model(self, texts, model, scaler):
        """使用sklearn模型进行预测"""
        features = extract_text_features(texts)
        features_scaled = scaler.transform(features)
        
        predictions = model.predict(features_scaled)
        
        return predictions

#训练sklearn模型
def train_sklearn_models(train_file, test_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_texts, train_labels = load_data(train_file)
    test_texts, test_labels = load_data(test_file)
    
    print("提取训练集特征...")
    train_features = extract_text_features(train_texts)
    print("提取开发集特征...")
    test_features = extract_text_features(test_texts)
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    joblib.dump(scaler, f"{output_dir}/feature_scaler.pkl")
    
    #训练随机森林模型
    print("训练随机森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(train_features_scaled, train_labels)
    
    #评估随机森林模型
    rf_preds = rf_model.predict(test_features_scaled)
    rf_f1 = f1_score(test_labels, rf_preds, average='macro')
    print(f"随机森林模型在测试集上的宏平均F1分数: {rf_f1:.4f}")
    joblib.dump(rf_model, f"{output_dir}/rf_model.pkl")
    
    #训练梯度提升模型
    print("训练梯度提升模型...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(train_features_scaled, train_labels)
    
    #评估梯度提升模型
    gb_preds = gb_model.predict(test_features_scaled)
    gb_f1 = f1_score(test_labels, gb_preds, average='macro')
    print(f"梯度提升模型在测试集上的宏平均F1分数: {gb_f1:.4f}")
    joblib.dump(gb_model, f"{output_dir}/gb_model.pkl")
    
    return rf_model, gb_model, scaler

def main(train_file, dev_file, test_file, output_file):
    
    
    models_config = [
        {
            'type': 'enhanced',
            'path': 'output/enhanced_model'
        },
        {
            'type': 'transformer',
            'path': 'output/simple_model/final_model'
        },
        {
            'type': 'random_forest',
            'path': './sklearn_models'
        },
        {
            'type': 'gradient_boosting',
            'path': './sklearn_models'
        }
    ]
    
    detector = EnsembleDetector(models_config)
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    scaler = joblib.load('./sklearn_models/feature_scaler.pkl')
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    test_texts = [item['text'] for item in test_data]
    
    if 'label' in test_data[0]:
        test_labels = [item['label'] for item in test_data]
    else:
        test_labels = None

    print("使用集成模型进行预测...")
    weights = [0.4, 0.3, 0.15, 0.15]  
    predictions = detector.predict(test_texts, weights=weights)
    
    if test_labels:
        print(f"\n=== 在集成模型中随机 100 条样本的集成评估 ===")
        print(classification_report(test_labels, predictions, digits=4))
        print("Macro F1:", f1_score(test_labels, predictions, average='macro'))

    results = []
    for i, pred in enumerate(predictions):
        result_item = {
            'text': test_data[i]['text'],
            'label': int(pred)
        }
        if 'id' in test_data[i]:
            result_item['id'] = test_data[i]['id']
        
        results.append(result_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"集成模型预测结果已保存到 {output_file}")

        # ====== 附加可视化 ======
    # 1. 加载开发集，并用各子模型及集成模型生成预测
    dev_texts, dev_labels = load_data(dev_file)
    # 利用各模型单独预测
    indiv_preds = []
    model_names = [cfg['type'] for cfg in models_config]
    for idx, (model, tok, scl, mtype) in enumerate(zip(detector.models, detector.tokenizers, detector.scalers, detector.model_types)):
        if mtype == 'enhanced':
            p = detector._predict_with_enhanced_model(dev_texts, model, tok, scl, batch_size=32)
        elif mtype == 'transformer':
            p = detector._predict_with_transformer_model(dev_texts, model, tok, batch_size=32)
        else:
            p = detector._predict_with_sklearn_model(dev_texts, model, scl)
        indiv_preds.append(p)
    # 集成预测
    ensemble_preds = detector.predict(dev_texts, weights=weights)

    # 2. 分类报告表格
    cr_dict = classification_report(dev_labels, ensemble_preds, output_dict=True, digits=4)
    df_cr = pd.DataFrame(cr_dict).T
    print("=== 集成模型分类报告 ===")
    display(df_cr)

    # 3. 混淆矩阵热力图
    cm = confusion_matrix(dev_labels, ensemble_preds)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Ensemble Confusion Matrix")
    plt.colorbar()
    ticks = sorted(set(dev_labels))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # 4. 各模型宏 F1 条形比较
    f1s = [f1_score(dev_labels, p, average='macro') for p in indiv_preds]
    f1s.append(f1_score(dev_labels, ensemble_preds, average='macro'))
    names = model_names + ['ensemble']
    plt.figure(figsize=(6,3))
    plt.bar(names, f1s, color='skyblue')
    plt.ylabel("Macro F1")
    plt.title("各模型 Macro F1 对比")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # 5. ROC 曲线对比
    try:
        plt.figure(figsize=(5,4))
        for preds, name in zip(indiv_preds + [ensemble_preds], names):
            fpr, tpr, _ = roc_curve(dev_labels, preds)
            auc = roc_auc_score(dev_labels, preds)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
        plt.plot([0,1], [0,1], linestyle='--', color='gray')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC 曲线对比")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    


    # 6. 特征分布直方图 & 箱型图
    feat_names = ['text_length','avg_sent_len','char_div','punc_ratio','digit_ratio','special_ratio','word_rep','ngram_ent']
    dev_feats = extract_text_features(dev_texts)
    dev_feats_scl = scaler.transform(dev_feats)
    df_feats = pd.DataFrame(dev_feats_scl, columns=feat_names)
    df_feats['label'] = dev_labels

    # 6a. 每个特征直方图
    for fn in feat_names:
        plt.figure(figsize=(4,2.5))
        plt.hist(df_feats[fn], bins=30)
        plt.title(f"Feature Distribution: {fn}")
        plt.xlabel(fn)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # 6b. 按类别箱型图
    for fn in feat_names:
        plt.figure(figsize=(4,2.5))
        df_feats.boxplot(column=fn, by='label')
        plt.title(fn + " by Label")
        plt.suptitle("")
        plt.xlabel("Label")
        plt.ylabel(fn)
        plt.tight_layout()
        plt.show()

    # 7. 特征相关性热力图
    corr = df_feats[feat_names].corr()
    plt.figure(figsize=(5,4))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(feat_names)), feat_names, rotation=45, ha='right')
    plt.yticks(range(len(feat_names)), feat_names)
    for i, j in np.ndindex(corr.shape):
        plt.text(j, i, f"{corr.iat[i,j]:.2f}", ha="center",
                 color="white" if abs(corr.iat[i,j])>0.5 else "black")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # 8. 随机森林 & GBDT 特征重要性
    try:
        rf_model = joblib.load('./sklearn_models/rf_model.pkl')
        gb_model = joblib.load('./sklearn_models/gb_model.pkl')
        fi_rf = rf_model.feature_importances_
        fi_gb = gb_model.feature_importances_
        x = np.arange(len(fi_rf))

        plt.figure(figsize=(5,2.5))
        plt.bar(x - 0.15, fi_rf, width=0.3, label='RF')
        plt.bar(x + 0.15, fi_gb, width=0.3, label='GB')
        plt.xticks(x, feat_names, rotation=45, ha='right')
        plt.ylabel("Importance")
        plt.title("Feature Importances")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass





if __name__ == "__main__":
    main(
        train_file="data/train.json",
        dev_file="data/dev.json",
        test_file="data/test_with_label.json",
        output_file="./ensemble_results.json"
    ) 

    