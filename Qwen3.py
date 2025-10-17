from unsloth import FastModel
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
import torch
def preparedata(dataset,tokenizer,base_prompt,text_col,label_col,alpaca_prompt):
     EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
     def formatting_prompts_func(examples):
         instruction = base_prompt  # single string, not a list
         inputs = examples[f"{text_col}"]
         outputs = examples[f"{label_col}"]
         texts = []
         for inp, out in zip(inputs, outputs):
           text = alpaca_prompt.format(instruction, inp, out) + EOS_TOKEN
           texts.append(text)
         return { "text" : texts, }
     return dataset.map(formatting_prompts_func, batched = True,)


from Workfile.Model import LgModel
class UnslothQwen(LgModel):
  def __init__(self,model_name,max_seq_length,Model = None,tokenizer = None,target_modules = None,r = 8):
    super().__init__(model_name,max_seq_length,Model,tokenizer)
    self.lora = target_modules
    self.r = r
  def preprocess(self,train_ds,test_ds,text_col,label_col):
     self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
     self.text_col = text_col
     self.label_col = label_col
     self.train_ds = preparedata(train_ds,self.tokenizer,self.instruction,text_col,label_col,self.alpaca_prompt)
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
  def inference(self,texts,max_seq_length):
    self.max_new_tokens = max_seq_length
    preds = []
    for text in tqdm(texts, desc="Evaluating"):
      prompt_text = self.alpaca_prompt.format(self.instruction, text, "") + self.tokenizer.eos_token
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

