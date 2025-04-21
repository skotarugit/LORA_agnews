# validate.py

import json
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class TextDataset(Dataset):
    """Simple torch Dataset for raw text & labels from HF dataset."""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # drop batch dim
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_model(model_path, device):
    """Load a PEFT-wrapped RoBERTa for seq classification."""
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    base = RobertaForSequenceClassification.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base, model_path)
    model.to(device).eval()
    return tokenizer, model

def run_validation(model_path, batch_size, max_length, output_json, output_plot):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer
    base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(class_names))
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    # Load AG News validation
    ds = load_dataset("ag_news")
    test_ds = stratified_split(dataset["train"], test_size=0.1)["validation"]
    texts = test_ds["test"]["text"]   # use test split as “validation”
    labels = test_ds["test"]["label"]

    # Build DataLoader
    if "input_ids" not in test_dataset.features:
        print("Preprocessing test dataset...")

        def preprocess_data(examples):
            return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

        processed_dataset = test_dataset.map(preprocess_data, batched=True, remove_columns=["text"])

        # Rename label column if needed
        if "label" in processed_dataset.features and "labels" not in processed_dataset.features:
            processed_dataset = processed_dataset.rename_column("label", "labels")
    else:
        processed_dataset = test_dataset

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Create dataloader
    loader = DataLoader(processed_dataset, batch_size=batch_size, collate_fn=data_collator)

    # Initialize containers
    all_preds = []
    all_labels = []
    all_texts = []
    all_probs = []

    # Evaluate
    print("Running evaluation...")

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits from dictionary
        logits = outputs.get("logits") if hasattr(outputs, "get") else outputs.logits
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(inputs["labels"].cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate per-class metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print evaluation results
    print("\n=== EVALUATION RESULTS ===")
    print(f"Total test samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")

    print("\nPer-class metrics:")
    for class_name in class_names:
        metrics = report[class_name]
        print(f"{class_name}: F1={metrics['f1-score']:.4f}, Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, Support={metrics['support']}")

    # Visualizations
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    print(f"Saved confusion matrix to {os.path.join(output_dir, 'confusion_matrix.png')}")

    # Plot per-class F1 scores
    plt.figure(figsize=(10, 6))
    class_f1 = [report[name]['f1-score'] for name in class_names]
    sns.barplot(x=class_names, y=class_f1)
    plt.title("F1 Score by Class")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_f1_scores.png"))
    print(f"Saved F1 score plot to {os.path.join(output_dir, 'class_f1_scores.png')}")

    # Plot precision and recall by class
    plt.figure(figsize=(12, 6))
    metrics_data = []
    for name in class_names:
        metrics_data.append({'Class': name, 'Metric': 'Precision', 'Value': report[name]['precision']})
        metrics_data.append({'Class': name, 'Metric': 'Recall', 'Value': report[name]['recall']})

    metrics_df = pd.DataFrame(metrics_data)
    sns.barplot(x='Class', y='Value', hue='Metric', data=metrics_df)
    plt.title("Precision and Recall by Class")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_by_class.png"))
    print(f"Saved precision/recall plot to {os.path.join(output_dir, 'precision_recall_by_class.png')}")

    # Save all results to JSON
    results = {
        "overall_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        },
        "class_metrics": report,
        "confusion_matrix": cm.tolist()
    }

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved complete evaluation results to {os.path.join(output_dir, 'evaluation_results.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./trained_models/results_lora/final_model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_json", type=str, default="val_metrics.json")
    parser.add_argument("--output_plot", type=str, default="confusion_matrix.png")
    args = parser.parse_args()

    run_validation(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_json=args.output_json,
        output_plot=args.output_plot
    )
