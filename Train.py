from TrainingLanguageModel.Utils import loadLoraModel,loadRandLoraModel,get_lora_modules
from peft import prepare_model_for_kbit_training
from TrainingLanguageModel.Utils import load_sequence_classification_model
from TrainingLanguageModel.NormalTransformer import SequenceClassification
from TrainingLanguageModel.CustomTrainer import apply_custom_loss
def training_SequenceClassification(workarg):

    # Safely extract with defaults
    model_name = workarg.get('model_name')
    max_seq    = workarg.get('max_seq', 64)
    num_labels = workarg.get('num_labels', 2)
    load_in_4bit = workarg.get('load_in_4bit', False)
    token = workarg.get('token', None)

    train_type = workarg.get('train_type', 'None')
    module     = workarg.get('module', None)
    rank       = workarg.get('rank', -1)
    imbalance_strategy = workarg.get('imbalance_strategy', None)

    train_arg  = workarg.get('train_arg', {})
    saveargs   = workarg.get('saveargs', None)

    train_ds   = workarg.get('train_ds')
    test_ds    = workarg.get('test_ds')

    text_col   = workarg.get('text_col', 'text')
    labels_col = workarg.get('labels_col', 'labels')

    # Load model + tokenizer
    model, tokenizer = load_sequence_classification_model(
        model_name,
        num_labels,
        token,
        load_in_4bit
    )

    # Lora modules (safe even if module=None) 
    target_modules = get_lora_modules(model, module) if module else ['Not Used']

    # Training type logic
    if train_type == 'Lora':
        model = loadLoraModel(model, target_modules, rank)

    elif train_type == 'RandLora':
        model = loadRandLoraModel(model, target_modules, rank)
        

    # Sequence manager (I assume bmodel == model and r == rank)
    sqm = SequenceClassification(
        model_name,
        max_seq,
        model,
        tokenizer,
        target_modules,
        rank
    )

    # Preprocess
    sqm.preprocess(train_ds, test_ds, text_col, labels_col)

    # Trainer
    sqm.prepare_trainer(train_arg)
    # Apply custom loss if specified
    apply_custom_loss(sqm, imbalance_strategy)
        

    # Train & test
    out = sqm.train_test(saveargs=saveargs)

    return out


from TrainingLanguageModel.Utils import PostProccsess
def training(train_ds,test_ds,point,upload = False,max_seq = 48,load_in_4bit = True,
             num_labels = 13,text_col = 'meta_description',labels_col = 'Category_id',imbalance_strategy = 'Not Used',alpha = 0.1
             ,saveargs = None):

    # Unpack for printing
    model_name = point['Model_name']
    t = point['t']
    target_modules = point['lora']
    r = point['r']

    # -----------------------------
    # Just print — no other changes
    # -----------------------------
    print("\n===== TRAINING CONFIG =====")
    print(f"Model_name     : {model_name}")
    print(f"Lora type (t)  : {t}")
    print(f"Target modules : {target_modules}")
    print(f"Rank (r)       : {r}")
    print(f"Upload mode    : {upload}")
    print(f"Max seq len    : {max_seq}")
    print(f"4-bit loading  : {load_in_4bit}")
    print(f"Num labels     : {num_labels}")
    print(f"Text column    : {text_col}")
    print(f"Labels column  : {labels_col}")
    print(f"Imbalance strat: {imbalance_strategy}")
    if imbalance_strategy == 'SmoothLabels':
        print(f"Alpha (SmoothLabels): {alpha}")
    if saveargs is not None:
     print(f"Save args      : {saveargs}")
    print("===========================\n")

    # -----------------------------
    # Your original code (unchanged)
    # -----------------------------
    Used_model, tokenizer = load_sequence_classification_model(
        model_name, num_labels, load_in_4bit=load_in_4bit
    )

    if upload:
        Used_model = prepare_model_for_kbit_training(
            Used_model, use_gradient_checkpointing=True
        )

    if t == 'lora':
        Used_model = loadLoraModel(Used_model, target_modules, r)
    elif t =='rlora':
        Used_model = loadRandLoraModel(Used_model, target_modules, r)
    else:
        Used_model = Used_model
    if upload:
        Used_model.gradient_checkpointing_enable()
    if t == 'rlora':
        if load_in_4bit:
         Used_model = Used_model.half()

    model = SequenceClassification(
        model_name, max_seq, Used_model, tokenizer, target_modules, r
    )

    model.preprocess(train_ds, test_ds, text_col, labels_col)
    model.prepare_trainer()
    apply_custom_loss(model, imbalance_strategy,alpha=alpha)
    out = model.train_test()
    if saveargs is not None:
     PostProccsess(out,saveargs)
    else:
     print("No save arguments provided, skipping post-processing.Shown the model performance")
     from TrainingLanguageModel.Utils import Evaluate_model
     evaluate = Evaluate_model()
     all_preds, all_labels = out['preds'], out['labels']
     evaluate.Clsevaluate(all_preds, all_labels, plot_cm=False)  
    return out