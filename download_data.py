import os
import pandas as pd

dataset_path = "data/saraga_raw/saraga1.5_hindustani"
output_csv = "data/saraga_metadata.csv"

print("Scanning directory structure directly...")
records = []
raga_mapping = {}
raga_counter = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_lower = file.lower()
        if ".mp3" in file_lower or ".wav" in file_lower or ".mp4" in file_lower:
            full_audio_path = os.path.join(root, file)
            
            # Use the folder name containing the audio file as the Raga Name
            raga_name = os.path.basename(root)
            
            if raga_name not in raga_mapping:
                raga_mapping[raga_name] = raga_counter
                raga_counter += 1
                
            records.append({
                "audio_path": full_audio_path,
                "raga_label": raga_mapping[raga_name]
            })

df_out = pd.DataFrame(records)
df_out.to_csv(output_csv, index=False)
print(f"Index built! Saved {len(df_out)} true audio files across {raga_counter} distinct ragas.")