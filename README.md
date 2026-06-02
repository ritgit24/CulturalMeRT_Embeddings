# Cross-Cultural Music Representation Evaluation: MERT vs. CultureMERT

## 1. Objective
The primary goal of this task is to evaluate and compare the musical intelligence of two state of the art self supervised audio models: **MERT** (a general Purpose Music Understanding Model) and **CultureMERT** (a multiple cultural adapted music understanding model). 

### The Why ?
* **The Limitation of Speech Models:** Traditional self-supervised audio frameworks (like HuBERT or Wav2Vec 2.0) are designed for spoken language and fail to capture complex polyphonic, tonal, and harmonic nuances in music.
* **The Domain Gap (MERT):** MERT addresses this by incorporating musical biases (like Constant-Q Transform representations) but is largely pre-trained on Western and contemporary commercial music distributions.
* **The Cultural Bias (CultureMERT):** Non-Western music traditions rely heavily on microtonal structures, non-standard tuning systems, and complex rhythmic meters. CultureMERT addresses this via a two-stage continual pre-training strategy to adapt MERT's layers to cross-cultural acoustic and modal features.

By evaluating both models on the **Saraga Dataset** (which features Hindustani and Carnatik classical music), we directly test CultureMERT’s culturally adapted representational space against standard MERT on complex modal structures (**Ragas**).


## 2. Outcomes of the project
* **Embedding Topology Analysis:** Observed how general-purpose vs. culturally-adapted self-supervised models cluster traditional microtonal music without task-specific training.
* **Fine-Tuning Adaptability:** Measured how performance scales when backbones are unfrozen and exposed to supervised optimization loops.
* **Hyperparameter Sensitivity:** Discovered how pooling strategies (Mean vs. Max pooling) preserve dense structural audio characteristics.


## 3. Project Structure
The repository is engineered to enforce modularity, dividing data pipelines, feature extractions, visualizations, and modeling scripts cleanly:

```text
MERTTASK/
├── data
├── src/
│   ├── dataset.py              # PyTorch Dataset handling 24kHz resampling, mono mixing, and chunking
│   ├── extract_embeddings.py   # Component extracting multi-layer representations via Hugging Face
│   ├── finetune.py             # Supervised PyTorch model wrapper with configurable layer freezing
│   ├── probe.py                # Downstream linear probing evaluator using Logistic Regression
│   └── visualize.py            # High-dimensional UMAP scatter plot generation script
├── download_data.py            # Dataset downloader and metadata assembly coordinator
├── main.py                     # Execution orchestrator testing hyperparameters sequentially
└── requirements.txt            # System dependencies manifest
```

---

## 4. Setup Instructions

### Step 1: Environment Initialization
Initialize a isolated Python virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate  
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Automated Data Ingestion
To execute the acquisition engine to grab the official MTG Saraga collection , run this command :
```bash
python download_data.py
```

### Step 3: Run Evaluation Engine
This command is used to trigger the pipeline execution block to generate embeddings, visualizations, probes, and fine-tuning runs:
```bash
python main.py
```

## 5. How I approached the problem 

### Theoretical Foundation
I tried to refer to the following research papers to develop foundations in MeRT:
* **MERT:** *MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training* ([Li et al., 2023](https://arxiv.org)) - Provided the foundational multi-teacher MLM framework combining acoustic and musical intelligence.
* **CultureMERT:** *CultureMERT: Cross-Cultural Music Understanding with Large-Scale Self-Supervised Training* ([ISMIR 2025](https://ismir.net)) -Established the two-stage continual pre-training strategy required to handle non-Western microtonal musical structures.
* **HuBERT:** *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units* ([Hsu et al., 2021](https://arxiv.org)) - Provided the structural blueprint for frame-level audio quantization.

---

### Connecting BERT to MERT (Core Analogy)
To solve the task of musical feature understanding, I mapped the text-processing mechanics of **BERT** directly onto the audio processing pipeline of **MERT**:


| Concept | BERT (Text) | MERT (Audio) |
| :--- | :--- | :--- |
| **Input Unit** | Text Words / Tokens | Raw Audio Frames (processed via CNN front-end) |
| **Core Architecture** | Transformer Encoder Only | Transformer Encoder Only |
| **Training Task** | **Masked Language Modeling (MLM):** Hides random words and predicts them using left-and-right text context. | **Masked Audio Modeling:** Hides random segments of audio frames and reconstructs them using surrounding musical context. |
