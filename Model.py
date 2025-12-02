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
    def extract_fields(self,keys: list, missing_value="Not have"):
     source = self.common_args.to_dict()
     result = {}
     for k in keys:
        v = source.get(k, None)
        result[k] = v if v is not None else missing_value
     result['n_gpu'] = self.trainer.args.n_gpu
     return result
    def train_test(self,saveargs = None):
        
        from TrainingLanguageModel.Utils import total_current_mem,total_peak_mem
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = total_current_mem()
        start_time = time.time()
        output = {}
        self.trainer.train()
        end_time = time.time()
        mem_peak = total_peak_mem()
        torch.cuda.empty_cache()
        self.mem_used_train = round(mem_peak - mem_before, 2)
        self.training_time = round(end_time - start_time, 2)
        preds,labels = self.test(self.max_seq_length)
        output['Model_name'] = self.model_name
        output['Train_size'] = len(self.train_ds)
        output['Test_size'] = len(self.test_ds)
        output['preds'] = preds
        output['labels'] = labels
        output['arg'] = self.extract_fields(self.essential_keys)
        output['lora'] = self.lora
        trainable_parameters, total_parameters = self.count_paramaters()
        output['Parameters'] = total_parameters
        output['Trainable_parameters'] = trainable_parameters
        output['r'] = self.r
        output['Memory Allocation'] = f"{self.mem_used_train}"
        output['Training Time'] = f"{self.training_time}"
        if saveargs is not None:
          self.save_modelHgface(saveargs)
        return output





