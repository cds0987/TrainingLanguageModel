from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig,RandLoraConfig, get_peft_model

def loadRawSequenceClassificationModel(model_name,num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    Model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        Model.resize_token_embeddings(len(tokenizer))
        Model.config.pad_token_id = tokenizer.pad_token_id
    return Model, tokenizer


def loadLoraModel(model, target_modules, r):
    config = LoraConfig(r=r,lora_alpha=r,target_modules = target_modules,lora_dropout = 0.05,bias = "none",task_type = "SEQ_CLS")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def loadRandLoraModel(model, target_modules, r):
    config = RandLoraConfig(r = r,lora_alpha = r,target_modules = target_modules)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model





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