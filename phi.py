import os

import torch
os.environ['HF_HOME'] = '/cs/labs/adiyoss/yuvalringel/cache/'

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from prompts import create_prompt
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

    return model.cuda(), tokenizer


def generate_results(args, model, tokenizer, dataset):
    results = []
    predicted_classes = []
    responses_failed_to_process = []
    for i in tqdm(range(dataset.num_rows)):
        # create prompt
        prompt = create_prompt(args.prompt, args.prompt_version, dataset[i]['Text'])

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
        return 'Neutral', failed_to_process
    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, nargs='+', choices=['phi-2', 'phi-3.5'], default='phi-3.5', help='Choose between model variants: "phi-2" for the smaller, lightweight model or "phi-3.5" for the slightly larger, more competitive model. Default is "phi-3.5".')
    parser.add_argument('--prompt', type=str, nargs='+', choices=['zero_shots', 'few_shots'], default='few_shots', help='Specify the type of prompting method: "zero_shots" for no examples or "few_shots" for examples in the prompt. Default is "few_shots".')
    parser.add_argument('--prompt-version', type=str, nargs='+', choices=['basic', 'comprehensive'], default='comprehensive', help='Choose the prompt version: "basic" for simple prompts or "comprehensive" for more detailed ones. Default is "comprehensive".')
    parser.add_argument('--output-dir', type=str, default='results/', help='Specify the directory where the results will be saved. Default is "results/".')
    parser.add_argument('--run-all', action='store_true', help='Use all choices for model, prompt, and prompt-version')
    args = parser.parse_args()


    if args.run_all:
        print("--run-all was passed. Overwriting the rest of the argumetns adn running all combinations.")
        args.model = ['phi-2', 'phi-3.5']
        args.prompt = ['zero_shots', 'few_shots']
        args.prompt_version = ['basic', 'comprehensive']


    print(args)

    file_path = 'data/hate_speech_data.xlsx'
    df = pd.read_excel(file_path)
    dataset = Dataset.from_pandas(df)

    models = args.model if isinstance(args.model, list) else [args.model]
    prompts = args.prompt if isinstance(args.prompt, list) else [args.prompt]
    versions = args.prompt_version if isinstance(args.prompt_version, list) else [args.prompt_version]

    for model in models:
        args.model = model
        model, tokenizer = get_model_and_tokenizer(args.model)
        for prompt in prompts:
            for version in versions:
                args.prompt = prompt
                args.prompt_version = version

                print(f"Evaluating {args.model} model with {args.prompt} prompts (Version: {args.prompt_version})...")

                args.results_dir = os.path.join(args.output_dir, f"{args.model}_{args.prompt_version}")

                args.results_file = os.path.join(args.results_dir, f"{args.prompt}.json")
                args.scores_file = os.path.join(args.results_dir, f"{args.prompt}_scores.json")
                
                generate_results(args, model, tokenizer, dataset)
                evaluate_results(args)

# python phi.py --model phi-2 --prompt zero_shots few_shots
# python phi.py --run-all