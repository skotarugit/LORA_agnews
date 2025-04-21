# test.py

import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel

class PickleDataset(Dataset):
    def __init__(self, pickle_path, tokenizer, max_length=512):
        with open(pickle_path, "rb") as f:
            self.items = pickle.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        text = rec.get("text", "")
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # squeeze the batch dim
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["idx"] = idx
        return enc

def generate_predictions_csv(
    model_path: str,
    test_pickle_path: str,
    output_csv_path: str,
    batch_size: int = 64,
    max_length: int = 512,
    device: str = None,
):
    # load tokenizer + model
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    base = RobertaForSequenceClassification.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base, model_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    # prepare data
    ds = PickleDataset(test_pickle_path, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)

    all_ids, all_preds = [], []
    for batch in tqdm(loader, desc="Predicting"):
        ids = batch.pop("idx")
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**inputs).logits
        preds = torch.argmax(out, dim=-1).cpu().numpy()
        all_ids.extend(ids if isinstance(ids, list) else ids.cpu().tolist())
        all_preds.extend(preds.tolist())

    df = pd.DataFrame({"id": all_ids, "prediction": all_preds})
    df.to_csv(output_csv_path, index=False)
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    default="./trained_models/results_lora/final_model")
    p.add_argument("--test_pickle",   default="test_unlabelled.pkl")
    p.add_argument("--output_csv",    default="predictions.csv")
    p.add_argument("--batch_size",    type=int, default=64)
    args = p.parse_args()

    df = generate_predictions_csv(
        model_path=args.model_path,
        test_pickle_path=args.test_pickle,
        output_csv_path=args.output_csv,
        batch_size=args.batch_size,
    )
    print(df.head())
