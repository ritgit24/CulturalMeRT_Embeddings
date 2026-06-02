import torch
from torch.utils.data import DataLoader, random_split
from dataset import SaragaRagaDataset
from extract_embeddings import extract_features
from visualize import plot_embedding_space
from probe import evaluate_linear_probe
from finetune import FineTuneMusicModel, train_model, evaluate_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_5s = SaragaRagaDataset("data/saraga_metadata.csv", clip_duration=5)
    dataset_10s = SaragaRagaDataset("data/saraga_metadata.csv", clip_duration=10)
    
    loader_5s = DataLoader(dataset_5s, batch_size=4, shuffle=False)
    
    models = {
        "MERT": "m-bain/MERT-base",
        "CultureMERT": "m-bain/CultureMERT-base"
    }
    
    pooling_strategies = ["mean", "max"]
    
    for model_name, model_path in models.items():
        for pool in pooling_strategies:
            emb, lbl = extract_features(model_path, loader_5s, device, pooling_strategy=pool)
            plot_embedding_space(emb, lbl, f"{model_name}_{pool}", f"{model_name} ({pool} pooling)")
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
