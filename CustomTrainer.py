import torch
import torch.nn as nn
from transformers import Trainer,TrainingArguments
import numpy as np
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none"
        )
        p = torch.exp(-ce_loss)
        loss = ((1 - p) ** self.gamma * ce_loss).mean()
        return loss


class CustomTrainer(Trainer):
    def __init__(self, *args, focal_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        device = logits.device
        if self.focal_loss is not None:
            if self.focal_loss.weight is not None:
                self.focal_loss.weight = self.focal_loss.weight.to(device)
            loss = self.focal_loss(logits, labels)
        else:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

        return (loss, outputs) if return_outputs else loss




class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=0.0):
        """
        samples_per_class: list or tensor of counts per class
        beta: hyperparameter for class balanced weight
        gamma: optional focal-style focusing parameter
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        samples_per_class = torch.tensor(samples_per_class, dtype=torch.float)
        cb_weights = (1 - beta) / (1 - beta ** samples_per_class)
        cb_weights = cb_weights / cb_weights.sum() * len(samples_per_class)
        self.register_buffer("cb_weights", cb_weights)
    def forward(self, logits, targets):
        weights = self.cb_weights.to(logits.device)
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=weights,
            reduction="none"
        )
        if self.gamma > 0:
            p = torch.exp(-ce_loss)
            loss = ((1 - p) ** self.gamma * ce_loss).mean()
        else:
            loss = ce_loss.mean()
        return loss


class CustomTrainerCB(CustomTrainer):
    def __init__(self, *args, class_balanced_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_balanced_loss = class_balanced_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_balanced_loss is not None:
            loss = self.class_balanced_loss(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return (loss, outputs) if return_outputs else loss


from torch.utils.data import DataLoader, WeightedRandomSampler

class ResampleTrainer(Trainer):
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        labels = np.array(train_dataset["labels"])
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

class CostSensitiveTrainer(Trainer):
    """
    Custom Trainer that applies cost-sensitive learning.
    Uses weighted CrossEntropyLoss based on class imbalance or custom cost matrix.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def set_FocalLoss(model):
  labels = np.array(model.train_ds["labels"])
  class_counts = np.bincount(labels)
  class_weights = 1.0 / class_counts
  class_weights = class_weights / class_weights.sum()
  class_weights = torch.tensor(class_weights, dtype=torch.float)
  focal_loss = FocalLoss(gamma=2, weight=class_weights)


  model.trainer = CustomTrainer(
    model = model.model,
    args = model.common_args,
    train_dataset = model.train_ds,
    processing_class = model.tokenizer,
    focal_loss = focal_loss
)
  

def set_ClassBalanced(model):
    labels = np.array(model.train_ds["labels"])
    class_counts = np.bincount(labels)  # number of samples per class
    samples_per_class = class_counts.tolist()
    # Initialize Class-Balanced Loss (optional gamma for focal-style)
    cb_loss = ClassBalancedLoss(samples_per_class, beta=0.9999, gamma=2.0)
    # Use CustomTrainerCB instead of original CustomTrainer
    model.trainer = CustomTrainerCB(
                         model=model.model,
                         args=model.common_args,
                         train_dataset=model.train_ds,
                         processing_class = model.tokenizer,
                         class_balanced_loss=cb_loss)
     

def set_Resample(model):
    model.trainer = ResampleTrainer(
                         model= model.model,
                         args = model.common_args,
                         train_dataset = model.train_ds,
                         tokenizer = model.tokenizer,
                         )
    

from sklearn.utils.class_weight import compute_class_weight
def set_CostSensitive(model):
   labels = np.array(model.train_ds["labels"])
   class_weights = compute_class_weight(
                         class_weight="balanced",
                         classes=np.unique(labels),
                         y=labels)
   class_weights = torch.tensor(class_weights, dtype=torch.float)         
   model.trainer = CostSensitiveTrainer(
                         model = model.model,
                         args = model.common_args,
                         tokenizer = model.tokenizer,
                         train_dataset = model.train_ds,
                         class_weights=class_weights  # 👈 key parameter
                         )
   
def set_SmoothLabels(model, alpha=0.1):
    model.common_args.label_smoothing_factor = 0.1
    model.trainer = Trainer(
                         model=model.model,
                         args=model.common_args,
                         train_dataset=model.train_ds,
                         tokenizer = model.tokenizer,
                         )
    
def apply_custom_loss(model, imbalanceclass_type):
    if imbalanceclass_type == "FocalLoss":
        set_FocalLoss(model)
    elif imbalanceclass_type == "ClassBalanced":
        set_ClassBalanced(model)
    elif imbalanceclass_type == "Resample":
        set_Resample(model)
    elif imbalanceclass_type == "CostSensitive":
        set_CostSensitive(model)
    elif imbalanceclass_type == "SmoothLabels":
        set_SmoothLabels(model, alpha=0.1)