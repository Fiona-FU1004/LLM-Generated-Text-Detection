import os
import json
import argparse
from time import time
from data_augmentation import augment_dataset, create_balanced_dataset
from llm_detector import train_and_evaluate as simple_train
from enhanced_detector import train_and_evaluate as enhanced_train
from ensemble_detector import main as ensemble_main

def main():
    parser = argparse.ArgumentParser(description='LLM文本检测器训练和预测')
    parser.add_argument('--stage', type=str, choices=['all', 'augment', 'train', 'predict'], default='all',
                        help='执行的阶段: all(全部), augment(数据增强), train(训练模型), predict(预测)')
    parser.add_argument('--model', type=str, choices=['simple', 'enhanced', 'ensemble'], default='ensemble',
                        help='使用的模型类型: simple(简单模型), enhanced(增强模型), ensemble(集成模型)')
    parser.add_argument('--train_file', type=str, default='data/train.json', help='训练数据文件路径')
    parser.add_argument('--dev_file', type=str, default='data/dev.json', help='开发集数据文件路径')
    parser.add_argument('--test_file', type=str, default='data/test_with_label.json', help='测试数据文件路径')
    parser.add_argument('--augment_factor', type=int, default=1, help='数据增强倍数')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    #记录开始时间
    start_time = time()
    
    if args.stage in ['all', 'augment']:
        print("=== 开始数据增强阶段 ===")
        augmented_dir = os.path.join(args.output_dir, 'augmented_data')
        os.makedirs(augmented_dir, exist_ok=True)
        
        augment_dataset(
            input_file=args.train_file,
            output_file=os.path.join(augmented_dir, 'train_augmented.json'),
            augment_factor=args.augment_factor
        )
        
        create_balanced_dataset(
            input_file=os.path.join(augmented_dir, 'train_augmented.json'),
            output_file=os.path.join(augmented_dir, 'train_balanced.json')
        )
        args.train_file = os.path.join(augmented_dir, 'train_balanced.json')
        print(f"数据增强完成，用时: {time() - start_time:.2f}秒")
    
    if args.stage in ['all', 'train']:
        print("=== 开始模型训练阶段 ===")
        train_start_time = time()
        
        if args.model == 'simple':
            print("训练简单模型...")
            model_dir = os.path.join(args.output_dir, 'simple_model')
            simple_train(
                train_file=args.train_file,
                dev_file=args.dev_file,
                output_dir=model_dir
            )
        
        elif args.model == 'enhanced':
            print("训练增强模型...")
            model_dir = os.path.join(args.output_dir, 'enhanced_model')
            enhanced_train(
                train_file=args.train_file,
                dev_file=args.dev_file,
                output_dir=model_dir
            )
        elif args.model == 'ensemble':
            print("训练所有模型（将使用集成）...")
            #训练简单模型
            simple_model_dir = os.path.join(args.output_dir, 'simple_model')
            simple_train(
                train_file=args.train_file,
                dev_file=args.dev_file,
                output_dir=simple_model_dir
            )
            #训练增强模型
            enhanced_model_dir = os.path.join(args.output_dir, 'enhanced_model')
            enhanced_train(
                train_file=args.train_file,
                dev_file=args.dev_file,
                output_dir=enhanced_model_dir
            )
        
        print(f"模型训练完成，用时: {time() - train_start_time:.2f}秒")
    
    if args.stage in ['all', 'predict']:
        print("=== 开始预测阶段 ===")
        predict_start_time = time()
        
        if args.model == 'ensemble':
            print("使用集成模型进行预测...")
            output_file = os.path.join(args.output_dir, 'ensemble_results.json')
            ensemble_main(
                train_file=args.train_file,
                dev_file=args.dev_file,
                test_file=args.test_file,
                output_file=output_file
            )
        
        elif args.model == 'enhanced':
            from enhanced_detector import predict_on_test_data
            
            print("使用增强模型进行预测...")
            model_dir = os.path.join(args.output_dir, 'enhanced_model')
            output_file = os.path.join(args.output_dir, 'enhanced_results.json')
            
            predict_on_test_data(
                test_file=args.test_file,
                model_dir=model_dir,
                output_file=output_file
            )
        
        elif args.model == 'simple':
            #导入预测函数
            from llm_detector import predict_on_test_data
            
            print("使用简单模型进行预测...")
            model_dir = os.path.join(args.output_dir, 'simple_model/final_model')
            output_file = os.path.join(args.output_dir, 'simple_results.json')
            
            predict_on_test_data(
                test_file=args.test_file,
                model_dir=model_dir,
                output_file=output_file
            )
        
        print(f"预测完成，用时: {time() - predict_start_time:.2f}秒")
    
    total_time = time() - start_time
    print(f"=== 完成所有任务，总用时: {total_time:.2f}秒 ({total_time/60:.2f}分钟) ===")

if __name__ == "__main__":
    main() 