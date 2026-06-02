import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_embedding_space(embeddings, labels, output_name, title_suffix):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels, palette="Spectral", style=labels, alpha=0.8)
    plt.title(f"Embedding Space Visualization - {title_suffix}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{output_name}.png")
    plt.close()
