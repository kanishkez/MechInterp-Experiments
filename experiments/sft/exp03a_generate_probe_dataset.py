import json
from datasets import load_dataset

def generate_probe_datasets():
    print("Loading datasets...")
    
    # Task 1: Python vs Java (250 each)
    # Self-generated loops and classes to represent basic control flow and definitions
    task1_data = []
    for i in range(250):
        # Python
        python_code = f"def function_py_{i}(data):\n    result = []\n    for item in data:\n        if item > {i}:\n            result.append(item)\n    return result\n"
        task1_data.append({"text": python_code, "label": 0, "class_name": "python"})
        
        # Java
        java_code = f"public List<Integer> functionJava{i}(List<Integer> data) {{\n    List<Integer> result = new ArrayList<>();\n    for (int item : data) {{\n        if (item > {i}) {{\n            result.add(item);\n        }}\n    }}\n    return result;\n}}\n"
        task1_data.append({"text": java_code, "label": 1, "class_name": "java"})
        
    # Task 2: Chat Template vs Raw (250 each)
    # We use Alpaca, same semantic content
    alpaca = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    
    task2_data = []
    count = 0
    for row in alpaca:
        if count >= 250: break
        raw = row["instruction"]
        template = f"<|im_start|>user\n{raw}<|im_end|>\n<|im_start|>assistant\n"
        
        task2_data.append({"text": raw[:500], "label": 0, "class_name": "raw"})
        task2_data.append({"text": template[:500], "label": 1, "class_name": "chat"})
        count += 1
            
    # Task 3: Instruction vs Continuation (250 each)
    # Instruction: Alpaca instruction
    # Continuation: Alpaca output
    
    task3_data = []
    count = 0
    for row in alpaca: # Continuing from stream
        if count >= 250: break
        instruction = row["instruction"]
        continuation = row["output"]
        
        task3_data.append({"text": instruction[:500], "label": 0, "class_name": "instruction"})
        task3_data.append({"text": continuation[:500], "label": 1, "class_name": "continuation"})
        count += 1
            
    # Task 4: Structured vs Unstructured (250 each)
    # Structured: Self-generated JSON/classes
    # Unstructured: Wikitext
    
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    task4_data = []
    for i in range(250):
        # Structured
        struct_code = f"class Configuration_{i}:\n    def __init__(self):\n        self.id = {i}\n        self.type = 'config'\n        self.metadata = {{'key': {i}}}\n"
        task4_data.append({"text": struct_code, "label": 0, "class_name": "structured"})
        
    count = 0
    for row in wiki:
        text = row["text"].strip()
        if len(text) > 50:
            if count >= 250: break
            task4_data.append({"text": text[:500], "label": 1, "class_name": "unstructured"})
            count += 1
            
    # Save all
    datasets = {
        "task1_lang": task1_data,
        "task2_chat": task2_data,
        "task3_inst": task3_data,
        "task4_struct": task4_data
    }
    
    with open("probe_datasets.json", "w") as f:
        json.dump(datasets, f, indent=2)
        
    print("Saved 4 datasets (500 items each) to probe_datasets.json")

if __name__ == "__main__":
    generate_probe_datasets()
