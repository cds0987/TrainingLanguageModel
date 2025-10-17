import pandas as pd
from huggingface_hub import login
from datasets import Dataset, load_dataset

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
          print(f"✅ Added row for model: {filtered_data.get('Model_name', 'Unknown')}")
        else:
          filtered_data = {k: v for k, v in data.items()if k not in ['preds', 'labels']}
          self.df = pd.DataFrame([filtered_data])
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