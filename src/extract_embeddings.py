import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import SaragaRagaDataset

def extract_features(model_name, dataloader, device, pooling_strategy="mean"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            outputs = model(waveforms, output_hidden_states=True)
            
            if pooling_strategy == "mean":
                embeddings = outputs.last_hidden_state.mean(dim=1)
            elif pooling_strategy == "max":
                embeddings = outputs.last_hidden_state.max(dim=1)[0]
            elif pooling_strategy == "concat_last4":
                embeddings = torch.cat([outputs.hidden_states[-i].mean(dim=1) for i in range(1, 5)], dim=-1)
                
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.vstack(all_embeddings), np.concatenate(all_labels)
