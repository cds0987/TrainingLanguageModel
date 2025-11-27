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
