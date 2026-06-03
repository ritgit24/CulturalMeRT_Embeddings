import torch
from torch.utils.data import Dataset
import pandas as pd
import soundfile as sf
import scipy.signal

class SaragaRagaDataset(Dataset):
    def __init__(self, metadata_csv, sample_rate=24000, clip_duration=5):
        self.df = pd.read_csv(metadata_csv)
        self.sample_rate = sample_rate
        self.clip_len = sample_rate * clip_duration
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        info = sf.info(row['audio_path'])
        native_sr = info.samplerate
        
        # Calculate exactly how many frames to read from disk (read 10 seconds to be safe)
        frames_to_read = int(native_sr * 10)
        
        data, sr = sf.read(row['audio_path'], frames=frames_to_read)
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        if sr != self.sample_rate:
            num_samples = int(len(data) * self.sample_rate / sr)
            data = scipy.signal.resample(data, num_samples)
            
        waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        
        if waveform.shape[1] >= self.clip_len:
            waveform = waveform[:, :self.clip_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_len - waveform.shape[1]))
            
        return waveform.squeeze(0), int(row['raga_label'])