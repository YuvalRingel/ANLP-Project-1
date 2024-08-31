import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
import Levenshtein
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import argparse

from visualization import extract_embeddings, plot_top_attention_wordcloud, visualize_embeddings


# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')


# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def preprocess_text(text, stop_words):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and short words
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

# Custom Dataset to handle our data correctly
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Define the compute_metrics function
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


# Function to merge subwords and refine attention scores
def merge_subwords(tokens):
    merged_tokens = []
    current_word = ""
    for token in tokens:
        token = re.sub(r'[^a-zA-Z]', '', token)
        if token.startswith("Ġ"):
            if current_word:
                merged_tokens.append(current_word)
            current_word = token[1:]
        else:
            current_word += token
    
    if current_word:
        merged_tokens.append(current_word)
    
    return merged_tokens

def refine_attention_scores(word_attention):
    refined_attention = defaultdict(float)
    for word, score in word_attention.items():
        if len(word) <= 3 and word not in ["idf", "IDF"]:
            continue

        merged = False
        for refined_word in list(refined_attention.keys()):
            if Levenshtein.distance(word, refined_word) <= 2:
                refined_attention[refined_word] += score
                merged = True
                break
        
        if not merged:
            merged_word = merge_subwords(word.split())
            refined_word = "".join(merged_word)

            if refined_word == "rael":
                refined_word = "israel"
            if refined_word in 'zionism':
                refined_word = 'zionism'
            if refined_word in 'antisemitism':
                refined_word = 'antisemitism'
            if refined_word in 'palestine' or refined_word in 'palestinian':
                refined_word = 'palestine'
            if "ps" in refined_word:
                refined_word = " 🍉"
            
            if refined_word:
                refined_attention[refined_word] += score
    
    return dict(refined_attention)



def main(args):
    # Load the data
    file_path = "./data/hate_speech_data.xlsx"
    data = pd.read_excel(file_path)

    # Filter out neutral sentiment (0)
    data = data[data['Label'] != 0]

    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare the data
    texts = data['Text'].tolist()
    labels = data['Label'].map({-1: 1, 1: 0}).tolist()  # Map sentiment to 0 and 1


    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate", clean_up_tokenization_spaces=False)
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-hate", num_labels=2)

    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)

    # Create the dataset using the custom dataset class
    dataset = CustomDataset(encodings, labels)

    # Split the dataset into training and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    # Create a data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        warmup_steps=5,
        weight_decay=0.01,
        learning_rate=0.0001,  # Adjust the learning rate here
        logging_dir='./logs',
        logging_steps=10,  # Log every 10 steps
        eval_strategy="epoch",  # Evaluate after every epoch
        save_strategy="epoch",  # Save the model after every epoch
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,  # Function to compute and return metrics
    )

    print("Evaluating before training...")
    pre_train_eval = trainer.evaluate()
    print("Pre-training evaluation:", pre_train_eval)

    print("Extracting embeddings before fine-tuning...")
    embeddings_before = extract_embeddings(texts, model, tokenizer)
    visualize_embeddings(embeddings_before, labels, "Embeddings Before Fine-Tuning", args.save_path)

    print("Fine-tuning the model...")
    trainer.train()

    print("Evaluating after training...")
    post_train_eval = trainer.evaluate()
    print("Post-training evaluation:", post_train_eval)

    print("Extracting embeddings after fine-tuning...")
    embeddings_after = extract_embeddings(texts, model, tokenizer)
    visualize_embeddings(embeddings_after, labels, "Embeddings After Fine-Tuning", args.save_path)

    # Analyze attention weights
    def get_attention_weights(model, inputs):
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1]
        return attentions.mean(dim=1).detach().cpu(), inputs['input_ids']


    # Create a DataLoader for the test set
    test_loader = DataLoader(test_dataset, batch_size=8)

    stop_words = set(stopwords.words('english'))
    word_attention = defaultdict(float)
    stemmer = PorterStemmer()

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            attention_weights, input_ids = get_attention_weights(model, {'input_ids': input_ids, 'attention_mask': attention_mask})
            
            for idx, (input_id, attn) in enumerate(zip(input_ids, attention_weights)):
                # Decode the entire sequence
                full_text = tokenizer.decode(input_id, skip_special_tokens=True)
                
                # Preprocess and tokenize the text
                tokens = preprocess_text(full_text, stop_words)
                
                # Calculate attention for each token
                token_attention = defaultdict(float)
                for token, attention in zip(tokens, attn.sum(dim=0)[:len(tokens)]):
                    stem = stemmer.stem(token)
                    token_attention[stem] += attention.item()
                
                # Update the global word_attention dictionary
                for stem, attn in token_attention.items():
                    word_attention[stem] += attn

    # Normalize the attention scores
    total_attention = sum(word_attention.values())
    word_attention = {k: v / total_attention for k, v in word_attention.items()}

    #stop_words = set(stopwords.words('english'))
    #word_attention = {k: v for k, v in word_attention.items() if k not in stop_words and len(k) > 1}



    # Use this function to visualize the top 100 words with highest attention scores
    plot_top_attention_wordcloud(word_attention, '.results/TweetEval', top_n=70)

    # Calculate the attention scores using the refined method
    refined_word_attention = refine_attention_scores(word_attention)

    # Normalize the refined attention scores
    total_attention = sum(refined_word_attention.values())
    refined_word_attention = {k: v / total_attention for k, v in refined_word_attention.items()}

    # Select top N words with highest attention
    top_words = dict(sorted(refined_word_attention.items(), key=lambda item: item[1], reverse=True)[:40])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TweetEval')
    parser.add_argument('--save_path', type=str, default='results/TweetEval')
    args = parser.parse_args()

    print(args)

    main(args)