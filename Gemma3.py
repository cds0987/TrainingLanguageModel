from unsloth import FastModel
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm

def convert_to_chatml(example, base_prompt,text_col,label_col):
    return {
        "conversations": [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": example[text_col]},
            {"role": "assistant", "content": example[label_col]},
        ]
    }

def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

def preparedata(dataset, base_prompt, tokenizer,text_col,label_col):
    dataset = dataset.map(lambda ex: convert_to_chatml(ex, base_prompt,text_col,label_col))
    dataset = dataset.map(lambda ex: formatting_prompts_func(ex, tokenizer), batched=True)
    return dataset

from TrainingLanguageModel.Model import LgModel
import torch
class UnslothGemma(LgModel):
  def __init__(self,model_name,max_seq_length,Model = None,tokenizer = None,target_modules = None,r = 8):
    super().__init__(model_name,max_seq_length,Model,tokenizer)
    self.lora = target_modules
    self.r = r
  def preprocess(self,train_ds,test_ds,base_prompt,text_col,label_col):
     self.text_col = text_col
     self.label_col = label_col
     self.instruction = base_prompt
     self.train_ds = preparedata(train_ds,self.instruction,self.tokenizer,text_col,label_col)
     self.test_ds  = test_ds
  def prepare_trainer(self,arg = None, mode="work"):
    default_args = {
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "warmup_steps": 5,
        "learning_rate": 2e-5,
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "output_dir": "outputs",
        "report_to": "none",
    }
    if arg:
        default_args.update(arg)
    if mode == "demo":
        default_args["max_steps"] = 3  # run only 3 training steps
        print("⚙️ Running in DEMO mode (max_steps=3)")
    else:
        print("🚀 Running in WORK mode (full training)")
    self.train_args = default_args
    self.trainer = SFTTrainer(
    model = self.model,
    tokenizer = self.tokenizer,
    train_dataset = self.train_ds,
    dataset_text_field = "text",
    max_seq_length = self.max_seq_length,
    args = SFTConfig(
     **default_args
    ),
)
    self.trainer = train_on_responses_only(self.trainer,instruction_part = "<start_of_turn>user\n",response_part = "<start_of_turn>model\n",)
  def inference(self,texts,max_seq_length):
    self.max_new_tokens = max_seq_length
    preds = []
    for text in tqdm(texts, desc="Evaluating"):
      prompt_text = self.instruction + text
    # Apply chat template
      chat_text = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
          # Tokenize
      inputs = self.tokenizer(chat_text, return_tensors="pt").to("cuda")
    # Generate
      with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature = self.temperature, top_p = self.top_p, top_k = self.top_k, # For non thinking
        )
      gen_text = self.tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()
      preds.append(gen_text)
    return preds
  def save_modelHgface(self,arg):
        username,modelname,token,savetype = arg['username'],arg['modelname'],arg['token'],arg['savetype']
        repo = f"{username}/{modelname}"
        if savetype == 'Lora':
           self.model.push_to_hub(f"{repo}", token=token, private=False)
           self.tokenizer.push_to_hub(f"{username}/{modelname}", token=token, private=False)
        elif savetype == 'merged_16bit':
           self.model.push_to_hub_merged(f"{repo}", self.tokenizer, save_method = "merged_16bit", token = token)