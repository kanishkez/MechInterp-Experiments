import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import pandas as pd

def evaluate_probes(activations_file, labels_dict):
    print(f"Loading {activations_file}...")
    data = np.load(activations_file)
    
    results = []
    
    for task_name, acts in data.items():
        # acts shape: [num_items, num_layers, d_model]
        labels = labels_dict[task_name]
        
        num_layers = acts.shape[1]
        
        print(f"Training probes for {task_name}...")
        for layer in range(num_layers):
            X = acts[:, layer, :]
            y = labels
            
            clf = LogisticRegression(max_iter=1000, random_state=42)
            scoring = ['accuracy', 'roc_auc', 'precision', 'recall']
            
            cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring, n_jobs=-1)
            
            res = {
                "task": task_name,
                "layer": layer,
                "accuracy": np.mean(cv_results['test_accuracy']),
                "auroc": np.mean(cv_results['test_roc_auc']),
                "precision": np.mean(cv_results['test_precision']),
                "recall": np.mean(cv_results['test_recall'])
            }
            results.append(res)
            
    return results

def main():
    # Load labels
    with open("probe_datasets.json") as f:
        datasets = json.load(f)
        
    labels_dict = {}
    for task_name, items in datasets.items():
        labels_dict[task_name] = np.array([item["label"] for item in items])
        
    print("Evaluating Base model...")
    base_results = evaluate_probes("activations_base.npz", labels_dict)
    
    print("Evaluating Instruct model...")
    instruct_results = evaluate_probes("activations_instruct.npz", labels_dict)
    
    for r in base_results: r["model"] = "Base"
    for r in instruct_results: r["model"] = "Instruct"
    
    df = pd.DataFrame(base_results + instruct_results)
    df.to_csv("probe_results.csv", index=False)
    print("Saved results to probe_results.csv")

if __name__ == "__main__":
    main()
