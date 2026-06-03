import os
os.environ["TORIO_USE_FFMPEG"] = "0"

import torchaudio
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
from src.dataset import SaragaRagaDataset
from src.extract_embeddings import extract_features
from src.visualize import plot_embedding_space
from src.probe import evaluate_linear_probe
from src.finetune import FineTuneMusicModel, train_model, evaluate_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_5s = SaragaRagaDataset("data/saraga_metadata.csv", clip_duration=5)
    
    df = dataset_5s.df
    class_counts = df['raga_label'].value_counts()
    
    valid_classes = class_counts[class_counts >= 3].index
    if len(df[df['raga_label'].isin(valid_classes)]) < 15:
        print("Dataset subset is too small for threshold >= 3. Falling back to use all available local files.")
        valid_classes = class_counts[class_counts >= 1].index
    
    filtered_df = df[df['raga_label'].isin(valid_classes)].reset_index(drop=True)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes)}
    filtered_df['raga_label'] = filtered_df['raga_label'].map(label_mapping)
    
    dataset_5s.df = filtered_df
    
    loader_5s = DataLoader(dataset_5s, batch_size=4, shuffle=False)
    
    models = {
        "MERT": "m-a-p/MERT-v1-95M",
        "CultureMERT": "ntua-slp/CultureMERT-95M"
    }
    
    pooling_strategies = ["mean", "max"]
    
    for model_name, model_path in models.items():
        for pool in pooling_strategies:
            emb, lbl = extract_features(model_path, loader_5s, device, pooling_strategy=pool)
            
            try:
                plot_embedding_space(emb, lbl, f"{model_name}_{pool}", f"{model_name} ({pool} pooling)")
            except Exception as e:
                print(f"Skipping visualization plot for {model_name}_{pool} due to low dataset size.")
                
            report = evaluate_linear_probe(emb, lbl)
            print(f"--- Linear Probe: {model_name} | Pooling: {pool} ---")
            print(report)
            
    train_size = int(0.8 * len(dataset_5s))
    test_size = len(dataset_5s) - train_size
    train_set, test_set = random_split(dataset_5s, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
    
    num_classes = len(dataset_5s.df['raga_label'].unique())
    
    for model_name, model_path in models.items():
        ft_model = FineTuneMusicModel(model_path, num_classes=num_classes, freeze_backbone=False, num_frozen_layers=6)
        train_model(ft_model, train_loader, epochs=3, lr=2e-5, device=device)
        ft_report = evaluate_model(ft_model, test_loader, device)
        print(f"--- Fine-tuned Supervised: {model_name} (6 layers frozen) ---")
        print(ft_report)

if __name__ == "__main__":
    main()