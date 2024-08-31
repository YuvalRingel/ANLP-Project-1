import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
import Levenshtein
import re
import numpy as np
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')


# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def preprocess_text(text):
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
    output_dir='./results',
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

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,  # Function to compute and return metrics
)

# Perform an evaluation before training
print("Evaluating before training...")
pre_train_eval = trainer.evaluate()
print("Pre-training evaluation:", pre_train_eval)

# Extract and visualize embeddings before fine-tuning
print("Extracting embeddings before fine-tuning...")
def extract_embeddings(texts, model, tokenizer):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=256)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        embeddings = hidden_states[-1][:, 0, :]  # Use the CLS token embeddings

    return embeddings.cpu().numpy()

def visualize_embeddings(embeddings, labels, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Use a discrete colormap for two classes: 0 and 1
    label_colors = {0: 'blue', 1: 'red'}
    colors = [label_colors[label] for label in labels]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, alpha=0.6, label=labels)
    
    # Add a legend with labels
    legend_labels = {0: 'Pro Israel', 1: 'Pro Palestine'}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[l], markerfacecolor=label_colors[l], markersize=10) for l in label_colors]
    plt.legend(handles=handles, title='Labels')

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()


embeddings_before = extract_embeddings(texts, model, tokenizer)
visualize_embeddings(embeddings_before, labels, "Embeddings Before Fine-Tuning")

# Fine-tune the model
print("Fine-tuning the model...")
trainer.train()

# Evaluate the model on the test set after training
print("Evaluating after training...")
post_train_eval = trainer.evaluate()
print("Post-training evaluation:", post_train_eval)

# Extract and visualize embeddings after fine-tuning
print("Extracting embeddings after fine-tuning...")
embeddings_after = extract_embeddings(texts, model, tokenizer)
visualize_embeddings(embeddings_after, labels, "Embeddings After Fine-Tuning")

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
            tokens = preprocess_text(full_text)
            
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

def compute_saliency_map(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    embedding_output = model.roberta.embeddings(input_ids=inputs['input_ids'])
    embedding_output.retain_grad()

    outputs = model(inputs_embeds=embedding_output, attention_mask=inputs['attention_mask'])
    loss = outputs.logits[0, outputs.logits.argmax()].backward()

    saliency = embedding_output.grad.abs().sum(dim=-1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    cleaned_tokens = [token.replace("Ġ", "") for token in tokens if token not in ["<s>", "</s>", "<pad>"]]

    saliency = saliency[:len(cleaned_tokens)]

    return cleaned_tokens, saliency

def plot_top_attention_wordcloud(word_attention, top_n=100, title="Top Attention Word Cloud"):
    top_words = dict(sorted(word_attention.items(), key=lambda item: item[1], reverse=True)[:top_n])
    
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          colormap='viridis',
                          min_font_size=10,
                          max_font_size=100,
                          random_state=42).generate_from_frequencies(top_words)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout(pad=0)
    plt.show()

# Use this function to visualize the top 100 words with highest attention scores
plot_top_attention_wordcloud(word_attention, top_n=70)

# Calculate the attention scores using the refined method
refined_word_attention = refine_attention_scores(word_attention)

# Normalize the refined attention scores
total_attention = sum(refined_word_attention.values())
refined_word_attention = {k: v / total_attention for k, v in refined_word_attention.items()}

# Select top N words with highest attention
top_words = dict(sorted(refined_word_attention.items(), key=lambda item: item[1], reverse=True)[:40])


def visualize_single_sentence_saliency(sentence, model, tokenizer):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [4, 1]})
    fig.suptitle("Saliency Analysis of Sentence", fontsize=20)

    cmap = LinearSegmentedColormap.from_list("", ["white", "red"])

    tokens, saliency = compute_saliency_map(sentence, model, tokenizer)
    saliency = saliency.cpu().numpy()
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    # Text highlighting
    ax_text = axs[0]
    ax_text.axis('off')
    ax_text.set_title("Highlighted Sentence", fontsize=18)

    words = sentence.split()
    n_words = len(words)
    words_per_line = 10
    n_lines = int(np.ceil(n_words / words_per_line))
    line_height = 1.0 / (n_lines + 1)
    word_spacing = 0.9 / words_per_line

    y_position = 0.9
    for i, word in enumerate(words):
        if i < len(saliency_norm):
            color = cmap(saliency_norm[i])
        else:
            color = cmap(0)
        
        x_position = 0.02 + (i % words_per_line) * word_spacing
        if i % words_per_line == 0 and i != 0:
            y_position -= line_height
        
        ax_text.text(x_position, y_position, word, ha='left', va='top',
                     bbox=dict(facecolor=color, edgecolor='none', alpha=0.8),
                     fontsize=14, transform=ax_text.transAxes)

    # Bar chart
    ax_bar = axs[1]
    ax_bar.set_title("Top 10 Salient Words", fontsize=16)
    
    sorted_indices = saliency.argsort()[::-1][:10]
    top_tokens = [tokens[i] for i in sorted_indices]
    top_saliency = saliency[sorted_indices]

    sns.barplot(x=top_saliency, y=top_tokens, ax=ax_bar, orient='h')
    ax_bar.set_xlabel("Saliency Score", fontsize=14)
    ax_bar.set_ylabel("Tokens", fontsize=14)

    plt.tight_layout(pad=2)
    plt.subplots_adjust(top=0.88)
    plt.show()
    
# Example usage
# single_sentence = "Footage from October 7th shows how Hamas treated the children, Israel will always lie about it."
# visualize_single_sentence_saliency(single_sentence, model, tokenizer)