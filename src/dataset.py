import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd

class SaragaRagaDataset(Dataset):
    def __init__(self, metadata_csv, sample_rate=24000, clip_duration=5):
        self.df = pd.read_csv(metadata_csv)
        self.sample_rate = sample_rate
        self.clip_len = sample_rate * clip_duration
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row['audio_path'])
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if waveform.shape[1] >= self.clip_len:
            waveform = waveform[:, :self.clip_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_len - waveform.shape[1]))
            
        return waveform.squeeze(0), int(row['raga_label'])
