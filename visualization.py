from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

def extract_embeddings(texts, model, tokenizer):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=256)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        embeddings = hidden_states[-1][:, 0, :]  # Use the CLS token embeddings

    return embeddings.cpu().numpy()


def visualize_embeddings(embeddings, labels, title, save_path):
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
    plt.savefig(save_path)


def plot_top_attention_wordcloud(word_attention, save_path, top_n=100, title="Top Attention Word Cloud"):
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
    plt.savefig(save_path)




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



def visualize_single_sentence_saliency(sentence, model, tokenizer, save_path):
    """
    Example usage
    single_sentence = "Footage from October 7th shows how Hamas treated the children, Israel will always lie about it."
    visualize_single_sentence_saliency(single_sentence, model, tokenizer)
    """
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

    plt.savefig(save_path)    
