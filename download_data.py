import os
import pandas as pd
import mirdata

data_directory = "data/saraga_raw"
os.makedirs(data_directory, exist_ok=True)

print("Downloading Saraga Carnatic Dataset")
carnatic_db = mirdata.initialize("saraga_carnatic", data_home=data_directory)
carnatic_db.download()

print("Downloading Saraga Hindustani Dataset")
hindustani_db = mirdata.initialize("saraga_hindustani", data_home=data_directory)
hindustani_db.download()

print("Building index mapping")
records = []
raga_mapping = {}
raga_counter = 0

all_tracks = {**carnatic_db.load_tracks(), **hindustani_db.load_tracks()}

for track_id, track_data in all_tracks.items():
    if hasattr(track_data, 'raga') and track_data.raga:
        raga_name = track_data.raga
        if raga_name not in raga_mapping:
            raga_mapping[raga_name] = raga_counter
            raga_counter += 1
            
        records.append({
            "audio_path": track_data.audio_path,
            "raga_label": raga_mapping[raga_name]
        })

df = pd.DataFrame(records)
df.to_csv("data/saraga_metadata.csv", index=False)
print(f"Dataset compiled successfully! Found {len(df)} tracks across {raga_counter} unique ragas.")
