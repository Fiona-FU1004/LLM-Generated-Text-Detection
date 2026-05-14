import json

def read_data_sample(file_path, n_samples=2):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples in {file_path}: {len(data)}")
    print(f"First {n_samples} samples:")
    for i, sample in enumerate(data[:n_samples]):
        print(f"\nSample {i+1}:")
        for key, value in sample.items():
            if key == "text":
                print(f"{key}: {value[:200]}...")
            else:
                print(f"{key}: {value}")

print("Development Dataset")
read_data_sample('data/dev.json')

print("\nTest Dataset")
read_data_sample('data/test.json')

print("\nTraining Dataset")
read_data_sample('data/train.json') 