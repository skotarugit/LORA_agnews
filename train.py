import logging
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    RobertaModel,
    RobertaPreTrainedModel,
    AutoConfig
)
from datasets import load_dataset, Dataset, ClassLabel
import evaluate
import numpy as np
from peft import LoraConfig, get_peft_model,PeftModel
from torch import nn
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
import nlpaug.augmenter.word as naw
import os
import random
from typing import Dict, List, Optional, Union, Any, Tuple

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration
# ---------------------------
class Config:
    base_model = "roberta-base"
    output_dir = "results_lora"
    use_fnn = True
    use_augmentation = True
    use_early_stopping = True
    use_weight_decay = True
    freeze_base_model = True
    early_stopping_patience = 3
    weight_decay_value = 0.01
    train_last_k_layers = 2
    max_seq_length = 512
    train_batch_size = 32
    eval_batch_size = 64
    num_train_epochs = 1
    learning_rate = 8e-6  
    # Class weights for loss function (higher weights for Business and Sci/Tech)
    class_weights = [1.0, 1.0, 1.0, 1.0]

    # LoRA Configuration - improved settings
    lora_r = 2  
    lora_alpha = 4  
    lora_dropout = 0.05  
    lora_bias = "none"
    lora_target_modules = ["query", "value"] 
    lora_task_type = "SEQ_CLS"

    # Seed for reproducibility
    seed = 42

# ---------------------------
# Set seeds for reproducibility
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------
# Custom Model Class
# ---------------------------
class RobertaWithClassifier(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)

        # Main classifier for all classes
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256,256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.num_labels)
        )

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        

# ---------------------------
# Custom Weighted Loss Trainer
# ---------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
            print(f"Using class weights: {self.class_weights}")
            self.class_weights = torch.tensor(self.class_weights).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        # Forward pass
        outputs = model(**inputs)
        
        # Get logits and labels
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        if logits is None:
            logits = outputs[1]
            
        if self.class_weights is not None and labels is not None:
            # Apply weighted loss
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        else:
            # Fall back to default loss or use the loss from the model outputs
            loss = outputs.get("loss", None)
            if loss is None:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


# ---------------------------
# Metrics Calculation
# ---------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    # Class-specific metrics
    class_f1 = f1_score(labels, predictions, average=None)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    # Add per-class F1 scores
    for i, class_score in enumerate(class_f1):
        metrics[f'f1_class_{i}'] = class_score

    return metrics

# ---------------------------
# Utility Functions
# ---------------------------
def preprocess_data(tokenizer, dataset, max_length):
    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
    return dataset.map(preprocess, batched=True, remove_columns=["text"])

def stratified_split(dataset, test_size=0.1, seed=42):
    """Split dataset while preserving the class distribution"""
    train_indices = []
    val_indices = []

    # Group by label - using 'label' column name instead of 'labels'
    label_to_indices = {}
    for i, label in enumerate(dataset['label']):  # Changed 'labels' to 'label'
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)

    # Sample for each class
    for label, indices in label_to_indices.items():
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split_idx])
        val_indices.extend(indices[split_idx:])

    return {
        'train': dataset.select(train_indices),
        'validation': dataset.select(val_indices)
    }


def augment_dataset(dataset):
    aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", top_k=5)
    train_dataset, test_dataset = stratified_split(dataset["train"], test_size=0.1)

    def augment_text(example):
        try:
            augmented = aug.augment(example["text"], n=1)
            return {"text": augmented[0] if isinstance(augmented, list) else augmented, "label": example["label"]}
        except Exception:
            return example

    # Filter sci/tech (3)
    subset = train_dataset.filter(lambda x: x["label"] in [3])

    # Apply augmentation
    augmented = subset.map(augment_text)

    untouched = train_dataset.filter(lambda x: x["label"] not in [3])
    full_augmented_train = Dataset.from_list(untouched.to_list() + subset.to_list() + augmented.to_list())

    return full_augmented_train, test_dataset


def freeze_model_parameters(model):
    print("Freezing base model parameters")
    for name, param in model.named_parameters():
        if "lora" not in name and "classifier" not in name:
            param.requires_grad = False

def evaluate_model(model, dataset, data_collator, device):
    """Perform a comprehensive evaluation with detailed metrics"""
    model.eval()
    loader = DataLoader(dataset, batch_size=64, collate_fn=data_collator)

    all_preds = []
    all_labels = []

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.get("logits") if hasattr(outputs, "get") else outputs.logits
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(inputs["labels"].cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Class-specific metrics
    report = classification_report(all_labels, all_preds, target_names=["World", "Sports", "Business", "Sci/Tech"], output_dict=True)

    print(f"Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    for class_name, metrics in report.items():
        if class_name in ["World", "Sports", "Business", "Sci/Tech"]:
            logger.info(f"  {class_name} - F1: {metrics['f1-score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

# ---------------------------
# Main Training Function
# ---------------------------
def train_model(config):
    # Set seed for reproducibility
    set_seed(config.seed)

    logger.info("Loading tokenizer and dataset")
    tokenizer = RobertaTokenizer.from_pretrained(config.base_model)
    tokenizer.model_max_length = config.max_seq_length

    dataset = load_dataset("ag_news")

    if config.use_augmentation:
        train_dataset, test_dataset = augment_dataset(dataset)
    else:
        split_datasets = stratified_split(dataset["train"], test_size=0.1)
        train_dataset, test_dataset = split_datasets['train'], split_datasets['validation']

    tokenized_train_dataset = preprocess_data(tokenizer, train_dataset, config.max_seq_length)
    tokenized_test_dataset = preprocess_data(tokenizer, test_dataset, config.max_seq_length)
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")

    num_labels = len(set(tokenized_train_dataset["labels"]))
    label_names = tokenized_train_dataset.features["labels"].names if isinstance(tokenized_train_dataset.features["labels"], ClassLabel) else ["World", "Sports", "Business", "Sci/Tech"]
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    logger.info(f"Class distribution in training set:")
    label_counts = np.bincount(tokenized_train_dataset["labels"])
    for idx, count in enumerate(label_counts):
        logger.info(f"  {id2label[idx]}: {count} examples ({count/len(tokenized_train_dataset)*100:.2f}%)")

    if config.use_fnn:
        model_config = AutoConfig.from_pretrained(config.base_model, num_labels=num_labels)
        model = RobertaWithClassifier.from_pretrained(config.base_model, config=model_config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(config.base_model, num_labels=num_labels, id2label=id2label, label2id=label2id)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.lora_target_modules,
        task_type=config.lora_task_type
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"LoRA configuration: {lora_config}")

    # Print trainable parameters info
    model.print_trainable_parameters()

    if config.freeze_base_model:
        freeze_model_parameters(model)

    training_args = TrainingArguments(
        output_dir=f'./trained_models/{config.output_dir}',
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=300,
        save_steps=900,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay_value if config.use_weight_decay else 0.0,
        logging_dir='./logs',
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # F1 for better handling of imbalance
        greater_is_better=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        report_to="wandb",
        optim="adamw_torch",
        fp16=True,  # Mixed precision for faster training
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)] if config.use_early_stopping else []

    logger.info("Initializing Weighted Loss Trainer")
    trainer = WeightedLossTrainer(
        class_weights=config.class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks
    )

    print("Starting training")
    trainer.train()

    print("Evaluating the model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run detailed evaluation
    eval_results = evaluate_model(model, tokenized_test_dataset, data_collator, device)

    # Regular evaluation with trainer
    trainer_results = trainer.evaluate()
    print(f"Trainer evaluation results: {trainer_results}")

    print("Saving the model and tokenizer")
    model.save_pretrained(f'./trained_models/{config.output_dir}/final_model')
    tokenizer.save_pretrained(f'./trained_models/{config.output_dir}/final_model')

    print("Script finished successfully")

    return eval_results

if __name__ == "__main__":
    config = Config()
    train_model(config)