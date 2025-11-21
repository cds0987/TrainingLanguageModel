import time
import torch
import pandas as pd
import gc
class LgModel:
    def __init__(self, model_name, max_seq_length,model = None,tokenizer = None):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = 0.7
        self.top_p = 0.8
        self.top_k = 20
        self.load_model(model,tokenizer)
    def load_model(self,Model = None,tokenizer = None):
      if Model  is  None:
        raise ValueError("You must provide model and tokenizer explicitly")
      else:
         self.model = Model
         self.tokenizer = tokenizer
    def preprocess(self, *args, **kwargs):
        pass
    def prepare_trainer(self, *args, **kwargs):
        pass
    def inference(self, text):
        pass
    def save_modelHgface(self, *args, **kwargs):
        pass
    def test(self,max_newtokens):
        texts = self.test_ds[self.text_col]
        preds = self.inference(texts,max_newtokens)
        labels = self.test_ds[self.label_col]
        return preds,labels[:len(preds)]
    def clear_memory(self, *args, **kwargs):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    def count_paramaters(self):
        try:
            trainable_params, total_params = self.model.get_nb_trainable_parameters()
        except AttributeError:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return trainable_params, total_params
    def train_test(self,saveargs = None):
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1e6
        start_time = time.time()
        output = {}
        self.trainer.train()
        end_time = time.time()
        mem_peak = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / 1e6
        self.mem_used_train = round(mem_peak - mem_before, 2)
        mem_inference_est = round(mem_after - mem_before, 2)
        self.training_time = round(end_time - start_time, 2)
        preds,labels = self.test(self.max_seq_length)
        output['Model_name'] = self.model_name
        output['Train_size'] = len(self.train_ds)
        output['Test_size'] = len(self.test_ds)
        output['preds'] = preds
        output['labels'] = labels
        output['arg'] = self.common_args.to_dict()
        output['lora'] = self.lora
        trainable_parameters, total_parameters = self.count_paramaters()
        output['Parameters'] = total_parameters
        output['Trainable_parameters'] = trainable_parameters
        output['r'] = self.r
        output['Memory Allocation'] = f"{self.mem_used_train}"
        output['Training Time'] = f"{self.training_time}"
        if saveargs is not None:
          self.save_modelHgface(saveargs)
        self.clear_memory()
        return output



import torch
def get_lora_modules(model):
    target_modules = set()
    ffd_modules = set()
    attention_modules = set()
    try:
      import bitsandbytes as bnb
      linear_classes = (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
      linear_classes = (torch.nn.Linear,)
    for name, module in model.named_modules():
      if isinstance(module, linear_classes):
        sub_name = name.split(".")[-1]
        lname = name.lower()
            # Add all linear layers to target_modules
        target_modules.add(sub_name)
            # Feed-forward layers are all except q, k, v
        if not any(ch in lname for ch in [ "q", "k", "v"]):
          ffd_modules.add(sub_name)
        else:
          attention_modules.add(sub_name)
    return sorted(ffd_modules), sorted(attention_modules),target_modules

