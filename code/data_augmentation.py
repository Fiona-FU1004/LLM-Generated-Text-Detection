import json
import random
import os
import numpy as np
from tqdm import tqdm
import jieba
import re
from sklearn.model_selection import train_test_split

#随机种子
random.seed(42)
np.random.seed(42)

#数据增强函数
class DataAugmentor:
    def __init__(self):
        #加载停用词表
        self.stopwords = set()
        try:
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    self.stopwords.add(line.strip())
        except:
            print("未找到停用词表文件，使用空的停用词表")
    
    def random_swap(self, words, n=1):
        """随机交换词语位置"""
        new_words = words.copy()
        length = len(new_words)
        if length <= 1:
            return new_words
        for _ in range(n):
            idx1, idx2 = random.sample(range(length), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words
    
    def random_deletion(self, words, p=0.1):
        """随机删除词语"""
        if len(words) <= 1:
            return words
        new_words = []
        for word in words:
            if word in self.stopwords or random.random() > p:
                new_words.append(word)
        #如果所有单词都被删除了，返回一个随机单词
        if len(new_words) == 0:
            return [random.choice(words)]
        return new_words
    
    def random_insertion(self, words, n=1):
        """随机插入词语"""
        new_words = words.copy()
        length = len(new_words)
        if length <= 1:
            return new_words
        for _ in range(n):
            #随机选择一个词语并在随机位置插入
            word_to_insert = random.choice(new_words)
            insert_pos = random.randint(0, len(new_words))
            new_words.insert(insert_pos, word_to_insert)
        return new_words
    
    def split_sentences(self, text):
        """分割文本为句子"""
        #按常见的中文句子结束符号分割
        sentences = re.split(r'[。！？；]', text)
        #过滤掉空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def truncate_text(self, text, max_length=300):
        """截断文本"""
        if len(text) <= max_length:
            return text
        #句子边界截断
        sentences = self.split_sentences(text)
        result = ""
        for sentence in sentences:
            if len(result) + len(sentence) + 1 <= max_length:
                result += sentence + "。"
            else:
                break
        #如果结果为空，eg.第一个句子超级长，则直接截断
        if not result:
            result = text[:max_length]
        
        return result
    
    def crop_text(self, text, min_ratio=0.7, max_ratio=0.9):
        """裁剪文本的一部分"""
        length = len(text)
        if length <= 30:  #很短的文本，不进行裁剪
            return text
        #随机选择裁剪比例
        ratio = random.uniform(min_ratio, max_ratio)
        new_length = int(length * ratio)
        #随机选择起始位置
        start = random.randint(0, length - new_length)
        return text[start:start+new_length]
    
    def synonym_replacement(self, text, n=2):
        """同义词替换（这里暂时先采用简单实现）"""
        synonyms = {
            '好': ['不错', '良好', '优秀', '棒'],
            '坏': ['不好', '糟糕', '差', '劣质'],
            '大': ['巨大', '庞大', '宏伟', '广阔'],
            '小': ['微小', '细小', '迷你', '袖珍'],
            '快': ['迅速', '敏捷', '急速', '快速'],
            '慢': ['缓慢', '迟缓', '慢吞吞', '缓行'],
            '多': ['众多', '许多', '大量', '丰富'],
            '少': ['稀少', '不多', '有限', '寥寥'],
            '美': ['漂亮', '好看', '美丽', '美好'],
            '丑': ['难看', '不堪', '丑陋', '不美']
        }
        words = list(jieba.cut(text))
        new_words = words.copy()
        
        #替换n个单词
        replacements_made = 0
        random.shuffle(words)

        for word in words:
            if word in synonyms and replacements_made < n:
                synonym = random.choice(synonyms[word])
                for i in range(len(new_words)):
                    if new_words[i] == word:
                        new_words[i] = synonym
                        replacements_made += 1
                        break
                        
                if replacements_made >= n:
                    break
        return ''.join(new_words)
    
    def augment_text(self, text, methods=None):
        """使用多种方法增强文本"""
        if methods is None:
            methods = ['swap', 'delete', 'insert', 'crop', 'truncate', 'synonym']
        
        #随机选择一种增强方法
        method = random.choice(methods)
        
        words = list(jieba.cut(text))
        
        if method == 'swap' and len(words) > 1:
            n_swaps = random.randint(1, min(3, len(words)//2))
            new_words = self.random_swap(words, n=n_swaps)
            return ''.join(new_words)
        
        elif method == 'delete':
            p_delete = random.uniform(0.05, 0.1)
            new_words = self.random_deletion(words, p=p_delete)
            return ''.join(new_words)
        
        elif method == 'insert' and len(words) > 1:
            n_inserts = random.randint(1, 3)
            new_words = self.random_insertion(words, n=n_inserts)
            return ''.join(new_words)
        
        elif method == 'crop':
            return self.crop_text(text)
        
        elif method == 'truncate':
            return self.truncate_text(text, max_length=random.randint(200, 500))
        
        elif method == 'synonym':
            n_replacements = random.randint(1, 3)
            return self.synonym_replacement(text, n=n_replacements)
        
        else:
            return text

#增强数据
def augment_dataset(input_file, output_file, augment_factor=1):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据样本数: {len(data)}")
    
    augmentor = DataAugmentor()
    
    augmented_data = []
    for item in tqdm(data, desc="增强数据"):
        augmented_data.append(item)
        
        text = item['text']
        label = item['label']
        
        for i in range(augment_factor):
            augmented_text = augmentor.augment_text(text)
            new_item = {
                'text': augmented_text,
                'label': label
            }
            for key, value in item.items():
                if key not in ['text', 'label']:
                    new_item[key] = value
            augmented_data.append(new_item)
    
    print(f"增强后的数据样本数: {len(augmented_data)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=4)
    
    print(f"增强后的数据已保存到 {output_file}")

#创建平衡的数据集
def create_balanced_dataset(input_file, output_file, max_samples_per_class=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    class_data = {}
    for item in data:
        label = item['label']
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(item)
    
    min_class_count = min(len(samples) for samples in class_data.values())
    
    samples_per_class = min_class_count if max_samples_per_class is None else min(min_class_count, max_samples_per_class)
    
    balanced_data = []
    for label, samples in class_data.items():
        selected_samples = random.sample(samples, samples_per_class)
        balanced_data.extend(selected_samples)
    
    print(f"原始数据样本数: {len(data)}")
    print(f"平衡后的数据样本数: {len(balanced_data)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=4)
    
    print(f"平衡后的数据已保存到 {output_file}")

def main():
    os.makedirs("augmented_data", exist_ok=True)
    
    print("增强训练数据...")
    augment_dataset(
        input_file="data/train.json",
        output_file="augmented_data/train_augmented.json",
        augment_factor=1  
    )
    
    print("创建平衡的训练数据集...")
    create_balanced_dataset(
        input_file="augmented_data/train_augmented.json",
        output_file="augmented_data/train_balanced.json"
    )

if __name__ == "__main__":
    main() 