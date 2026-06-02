import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import DataLoader

class FineTuneMusicModel(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=False, num_frozen_layers=0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif num_frozen_layers > 0:
            for param in self.backbone.feature_extractor.parameters():
                param.requires_grad = False
            for layer in self.backbone.encoder.layers[:num_frozen_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
        
    def forward(self, x):
        outputs = self.backbone(x)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(embeddings)

def train_model(model, dataloader, epochs, lr, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for waveforms, labels in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
def evaluate_model(model, dataloader, device):
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            outputs = model(waveforms)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    from sklearn.metrics import classification_report
    return classification_report(all_labels, all_preds)
