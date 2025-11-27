from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import torch

def load_sequence_classification_model(
    model_name: str,
    num_labels: int,
    token: None = None,
    load_in_4bit: bool = False,
    compute_dtype: torch.dtype = torch.float16,
):
    # --- Tokenizer kwargs ---
    tok_kwargs = {}
    if token is not None:
        tok_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        if tokenizer.pad_token == "[PAD]":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # --- Optional 4-bit config ---
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=compute_dtype,
        )

    # --- Model kwargs ---
    model_kwargs = {
        "quantization_config": quant_config,
        "num_labels": num_labels,
        "ignore_mismatched_sizes": True,
    }
    if token is not None:
        model_kwargs["token"] = token

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **model_kwargs,
    )

    # Resize after adding pad token
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


import torch

def get_lora_modules(model,type = 'att'):
    target_modules = set()
    ffd_modules = set()
    attention_modules = set()

    # Detect all supported linear classes (Linear, Linear4bit, Linear8bitLt)
    try:
        import bitsandbytes as bnb
        linear_classes = (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        linear_classes = (torch.nn.Linear,)

    # ---- Pass 1: Identify all linear layers in order ----
    linear_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, linear_classes):
            linear_layer_names.append(name)

    # If no linear layers found, return empty sets
    if not linear_layer_names:
        return [], [], []

    # The last layer is simply the last linear module in traversal order
    last_linear_layer = linear_layer_names[-1].lower()

    # ---- Pass 2: Classify layers + exclude last layer ----
    for name, module in model.named_modules():
        lname = name.lower()

        # Skip the last linear layer dynamically
        if lname == last_linear_layer:
            continue

        if isinstance(module, linear_classes):
            sub = name.split(".")[-1]

            # Record all linear layers
            target_modules.add(sub)

            # Attention: q, k, v layers
            if any(att in lname for att in ["q", "k", "v"]):
                attention_modules.add(sub)
            else:
                ffd_modules.add(sub)
    if type == 'all':
        modules = sorted(target_modules)
    elif type == 'att':
        modules = sorted(attention_modules)
    else:
        modules = sorted(ffd_modules)
    return modules


from peft import get_peft_model, LoraConfig, RandLoraConfig
def loadLoraModel(model, target_modules, r):
    config = LoraConfig(r=r,lora_alpha=r,target_modules = target_modules,lora_dropout = 0.05,bias = "none",task_type = "SEQ_CLS")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def loadRandLoraModel(model, target_modules, r):
    config = RandLoraConfig(r = r,target_modules  = target_modules )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def total_current_mem():
    """Sum current memory across all GPUs (in MB)."""
    total = 0
    for i in range(torch.cuda.device_count()):
            total += torch.cuda.memory_allocated(i)
    return total / 1e6


def total_peak_mem():
    """Sum peak memory across all GPUs (in MB)."""
    total = 0
    for i in range(torch.cuda.device_count()):
            total += torch.cuda.max_memory_allocated(i)
    return total / 1e6

from datasets import load_dataset, ClassLabel

def load_and_prepare_dataset(
    dataset_name: str,
    label_column: str = "Category_id",
    cast_labels: bool = True,
    train_split: float = None,
    test_split: float = None,
    stratify: bool = True,
):
    """
    Load a HuggingFace dataset and optionally:
      • cast labels into ClassLabel
      • perform stratified or random splits on train/test sets

    Parameters
    ----------
    dataset_name : str
        Name of the dataset on HuggingFace Hub.
    label_column : str
        Column name that holds class ids.
    cast_labels : bool
        Convert to ClassLabel (required for some models).
    train_split : float
        Fraction to keep for train from the original dataset["train"].
        Example: train_split=0.25 keeps 25% of the train split.
    test_split : float
        Same logic for dataset["test"].
    stratify : bool
        If True, use stratified split on label_column.

    Returns
    -------
    train_ds, test_ds
    """

    # Load dataset
    ds = load_dataset(dataset_name)

    # Optionally cast label column to ClassLabel
    if cast_labels:
        num_classes = len(set(ds["train"][label_column]))
        ds = ds.cast_column(
            label_column,
            ClassLabel(num_classes=num_classes)
        )

    # Process train split
    if train_split is not None:
        train_ds = ds["train"].train_test_split(
            test_size=train_split,
            stratify_by_column=label_column if stratify else None
        )["train"]
    else:
        train_ds = ds["train"]

    # Process test split
    if test_split is not None:
        test_ds = ds["test"].train_test_split(
            test_size=test_split,
            stratify_by_column=label_column if stratify else None
        )["train"]
    else:
        test_ds = ds["test"]

    return train_ds, test_ds



import pandas as pd
from huggingface_hub import login
from datasets import Dataset, load_dataset
def performance_proccess(df):
  if "Performance" not in df.columns:
      return df
  perf = pd.json_normalize(df['Performance'])
  perf.columns = [f'{col}' for col in perf.columns]
  df = pd.concat([df, perf], axis=1)
  perf_cols = ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']
  df[perf_cols] = df[perf_cols] * 100
  df = df.drop(columns=['Performance','arg','Parameters'])
  return df
class Database:
    def __init__(self, token, username, dataname, columns=None):

        self.main_use = True
        self.Login(token, username, dataname)

    def pushdata_to_hgface(self):
        """Push DataFrame to Hugging Face Hub as a dataset."""
        if self.token is None:
            print("❌ Cannot push: Token missing. Please login first using a valid token.")
            return

        dataset = Dataset.from_pandas(self.df)
        dataset.push_to_hub(
            f"{self.username}/{self.data_name}",
            private=False,
            token=self.token
        )
        print(f"✅ Successfully pushed {len(self.df)} records to {self.username}/{self.data_name}")

    def Login(self, token, username, dataname):
        """Authenticate and initialize dataset."""
        self.token = token
        self.username = username
        self.data_name = dataname
        self.init_fetchDatabase()

    def init_fetchDatabase(self):
        """Fetch existing dataset and return a DataFrame."""
        try:
            dataset = load_dataset(f"{self.username}/{self.data_name}", split="train")
            df = pd.DataFrame(dataset)
            df = performance_proccess(df)
            self.columns = list(df.columns)
        except Exception as e:
            self.columns = None
            df = None
        self.df = df

    # 🆕 NEW FUNCTION
    def expand_columns(self, new_data: dict):
        """Add new columns dynamically if they are missing from the DataFrame."""
        new_cols = [k for k in new_data.keys() if k not in self.df.columns]
        for col in new_cols:
            print(f"🆕 Adding new column: {col}")
            self.df[col] = None
            if col not in self.columns:
                self.columns.append(col)
    def expand_columns(self, new_data: dict):
     if self.columns is not None:
        for col, value in new_data.items():
            new_col = col
            suffix = 1
            # Find a unique column name
            while new_col in self.df.columns:
                new_col = f"{col}_{suffix}"
                suffix += 1
            print(f"🆕 Adding new column: {new_col}")
            self.df[new_col] = value
            if new_col not in self.columns:
                self.columns.append(new_col)
     else:
        self.df = pd.DataFrame(new_data)
        self.columns = list(self.df.columns)

    def update_or_add_row(self, data: dict):
        """Add or update a row using only allowed columns."""
        if self.df is not None:
          filtered_data = {k: v for k, v in data.items() if k in self.df.columns}
          new_row = pd.DataFrame([filtered_data])
          self.df = pd.concat([self.df, new_row], ignore_index=True)
          self.df = performance_proccess(self.df)
          print(f"✅ Added row for model: {filtered_data.get('Model_name', 'Unknown')}")
        else:
          filtered_data = {k: v for k, v in data.items()if k not in ['preds', 'labels']}
          self.df = pd.DataFrame([filtered_data])
          self.df = performance_proccess(self.df)
          print(f"Added first")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluate_model:
  def __init__(self,):
    pass
  def Clsevaluate(self,all_preds,all_labels, plot_cm=True):
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    # Confusion matrix
    if plot_cm:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=False, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall
    }
    

def get_top3_lora(ds):
    df = ds.copy()
    # 1. normalize lora to tuples (so duplicates are hashable)
    df["lora"] = df["lora"].apply(
        lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x
    )
    # 2. for each (Model_name, lora) keep row with best accuracy
    best_unique = df.loc[
        df.groupby(["Model_name", "lora"])["accuracy"].idxmax()
    ]
    # 3. for each model keep top-3 accuracy
    top3 = (
        best_unique.sort_values(["Model_name", "accuracy"], ascending=[True, False])
                  .groupby("Model_name")
                  .head(3)
                  .reset_index(drop=True)
    )
    return top3
def agreegate(df):
  best_per_model = df.loc[df.groupby("Model_name")["accuracy"].idxmax()].reset_index(drop=True)
  return best_per_model


def PostProccsess(output, args, categorymap=None):
    user_name, Main_name = args['username'], args['DataConfig']
    evaluate = Evaluate_model()

    # --- convert category text to numeric if categorymap is provided ---
    if categorymap is not None:
        # Reverse mapping: {"Healthcare": 6, ...}
        reverse_map = {v: k for k, v in categorymap.items()}

        def convert_to_numeric(x):
            if isinstance(x, str):
                return reverse_map.get(x, x)
            elif isinstance(x, list):
                return [reverse_map.get(i, i) for i in x]
            else:
                return x

        output['preds'] = convert_to_numeric(output['preds'])
        output['labels'] = convert_to_numeric(output['labels'])

    # --- evaluate performance ---
    all_preds, all_labels = output['preds'], output['labels']
    Performance = evaluate.Clsevaluate(all_preds, all_labels, plot_cm=False)
    for k, v in Performance.items():
      output[k] = v
    # --- main database ---
    MainDatabase = Database(args['token'], user_name, Main_name)
    MainDatabase.update_or_add_row(output)
    MainDatabase.pushdata_to_hgface()