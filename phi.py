import os

import torch
os.environ['HF_HOME'] = '/cs/labs/adiyoss/yuvalringel/cache/'

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from visualization import extract_embeddings, visualize_embeddings
from prompts import create_prompt, create_exercise_sample_prompt
import json

LABEL_TO_CLASS = {
    1: "Pro-Israeli",
    -1: "Pro-Palestinian",
    0: "Neutral"
}

CLASS_TO_LABEL = {
    "Pro-Israeli": 1,
    "Pro-Palestinian": -1,
    "Neutral": 0
}

def label_to_class(label):
    try:
        return LABEL_TO_CLASS[label]
    except KeyError:
        raise ValueError(f"Invalid label: {label}")

def class_to_label(cls):
    try:
        return CLASS_TO_LABEL[cls]
    except KeyError:
        raise ValueError(f"Invalid class: {cls}")

def get_model_and_tokenizer(model_name):
    # load model and tokenizer
    if args.model == 'phi-2':
        model_name = "microsoft/phi-2"
    elif args.model == 'phi-3.5':
        model_name = "microsoft/Phi-3.5-mini-instruct"
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if args.model == 'phi-2':
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        )

    return model, tokenizer


def generate_results(args, model, tokenizer, dataset):
    results = []
    predicted_classes = []
    responses_failed_to_process = []
    for i in tqdm(range(dataset.num_rows)):
        # create prompt
        prompt = create_prompt(args.prompt, dataset[i]['Text'])

        # encode
        inputs = {k: v.cuda() for k, v in tokenizer(prompt, return_tensors="pt", return_attention_mask=False).items()}
        prompt_encodings_length = inputs['input_ids'].shape[1]

        # generate
        outputs = model.generate(**inputs, max_length=prompt_encodings_length+10, pad_token_id=tokenizer.pad_token_id)

        # decode
        text = tokenizer.batch_decode(outputs)[0]

        # extract classification
        predicted_class, failed_to_process = extract_class_from_response(text)
        predicted_classes.append(predicted_class)
        responses_failed_to_process.append(failed_to_process)
        results.append({
            'row': i,
            'text': dataset[i]['Text'],
            'label': dataset[i]['Label'],
            'class': label_to_class(dataset[i]['Label']),
            'prompt': prompt,
            'llm_response': text,
            'predicted_label': class_to_label(predicted_class),
            'predicted_class': predicted_class
        })
    print(f"Failed to process {sum(responses_failed_to_process)} out of {dataset.num_rows} responses... considering them neutral..")
    # save results as json
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=4)


def evaluate_results(args):
    # load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    labels = [result['label'] for result in results]
    predictied_labels = [result['predicted_label'] for result in results]
    scores = compute_metrics_for_json(labels, predictied_labels)
    print(json.dumps(scores, indent=4))

    # save results as json
    with open(args.scores_file, 'w') as f:
        json.dump(scores, f, indent=4)


def extract_class_from_response(response):
    # response = response.removeprefix(prompt).removesuffix('<|endoftext|>').strip()
    response = response.split('Output:\nThis statement is')[-1].removesuffix('<|endoftext|>').strip()
    
    pro_israeli = 'pro-israeli'.lower() in response.lower()
    pro_palestinian = 'pro-palestinian' in response.lower()
    neutral = 'neutral' in response.lower()
    failed_to_process = False
    
    # if more than one class is mentioned in the response, return the response as is
    if sum([pro_israeli, pro_palestinian, neutral]) > 1:
        return response, failed_to_process
    
    # if no class found, return Neutral
    if sum([pro_israeli, pro_palestinian, neutral]) == 0:
        neutral = True
        failed_to_process = True

    
    return label_to_class(int(pro_israeli) - int(pro_palestinian)), failed_to_process


def compute_metrics_for_json(labels, preds):
    accuracy = accuracy_score(labels, preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        np.array(labels), np.array(preds), average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def finetune_model(model, tokenizer, train_dataset, test_dataset):
    def get_dataset_for_finetuning(dataset):
        dataset_prompts = [create_exercise_sample_prompt(dataset[i]['Text'], label_to_class(dataset[i]['Label'])) for i in range(len(dataset))]
        encodings = tokenizer(dataset_prompts, truncation=True)
        encodings.data['labels'] = [input_ids[1:] + input_ids[:1] for input_ids in encodings['input_ids']] 
        return Dataset.from_dict(encodings)
    
    train_dataset = get_dataset_for_finetuning(train_dataset)
    test_dataset = get_dataset_for_finetuning(test_dataset)

    response_template = " ## Output:\nThis statement is "
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=5,
        weight_decay=0.01,
        learning_rate=0.0001,
        logging_dir='./logs',
        logging_steps=10, 
        eval_strategy="epoch", 
        save_strategy="epoch",
    )
    def formatting_prompts_func(example):
        return example

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        formatting_func=formatting_prompts_func,
        data_collator=collator
    )

    # # Perform an evaluation before training
    # print("Evaluating before training...")
    # pre_train_eval = trainer.evaluate()
    # print("Pre-training evaluation:", pre_train_eval)

    # # Analyze the model's embeddings before fine-tuning
    # texts = [train_dataset[i]['Text'] for i in range(len(train_dataset))]
    # labels = [train_dataset[i]['Label'] for i in range(len(train_dataset))]
    
    # print("Extracting embeddings before fine-tuning...")
    # embeddings = extract_embeddings(texts, model, tokenizer)
    # visualize_embeddings(embeddings, labels, "Embeddings before fine-tuning")
    
    print("Fine-tuning the model...")
    trainer.train()

    # print("Evaluating after training...")
    # post_train_eval = trainer.evaluate()
    # print("Post-training evaluation:", post_train_eval)

    # print("Extracting embeddings after fine-tuning...")
    # embeddings_after = extract_embeddings(texts, model, tokenizer)
    # visualize_embeddings(embeddings_after, labels, "Embeddings After Fine-Tuning")
    
    finetuned_model = trainer.model
    finetuned_model.eval()
    finetuned_model.save_pretrained(f"./results/finetuned_{args.model}")
    return finetuned_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['phi-2', 'phi-3.5'])
    parser.add_argument('--prompt', type=str, nargs='+', choices=['zero_shots', 'few_shots'])
    parser.add_argument('--results-dir', type=str, default='results/')
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()

    partial_path = f"{args.model}_{args.finetune}" if args.finetune else args.model
    args.results_dir = os.path.join(args.results_dir, partial_path)

    print(args)

    # load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model)

    file_path = 'data/hate_speech_data.xlsx'
    df = pd.read_excel(file_path)

    dataset = Dataset.from_pandas(df)
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    if args.finetune:
        model = finetune_model(model, tokenizer, train_dataset, test_dataset)

    prompts = args.prompt
    for prompt in prompts:
        args.prompt = prompt
        args.results_file = os.path.join(args.results_dir, f"{args.prompt}.json")
        args.scores_file = os.path.join(args.results_dir, f"{args.prompt}_scores.json")
        
        print(f"Evaluating {args.model} model with {args.prompt} prompts")

        generate_results(args, model, tokenizer, test_dataset)
        evaluate_results(args)


# Example:
# python phi.py --model phi-2 --prompt zero_shots few_shots --finetune