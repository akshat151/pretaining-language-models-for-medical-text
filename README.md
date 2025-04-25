# Improving Diagnosis Prediction with Domain-Specific Pretraining

This repository explores enhancing clinical NLP using domain-adapted pretraining and task-specific fine-tuning. The project uses the [**MeDAL**](https://www.kaggle.com/datasets/xhlulu/medal-emnlp) dataset for pretraining and [**MIMIC-IV Notes**](https://physionet.org/content/mimiciii-demo/1.4/) for fine-tuning. We evaluate LSTM, LSTM + Self-Attention, and Transformer architectures in stage 1, and fine-tune the pretrained LSTM, LSTM + Self-Attention for stage 2.

📄 Prior Work: [MeDAL Paper](https://arxiv.org/pdf/2012.13978)

---

## Project Overview

### **STEP 1: Pretraining on MeDAL (Medical Abbreviation Disambiguation)**

- Goal: Equip models with medical contextual understanding.
- Dataset: MeDAL  
- Task: **Multi-class** abbreviation disambiguation with 22,555 classes
- Notebook References:
    - LSTM:
        - `notebooks/medal_lstm_demo.ipynb`
    - LSTM + Self Attention: 
        - `notebooks/medal_50_ctx_lstm_attention_demo.ipynb`
        - `notebooks/medal_100_ctx_lstm_attention_demo.ipynb`
        - `notebooks/medal_200_ctx_lstm_attention_demo.ipynb`
        - `notebooks/medal_lstm_selfattention_v2.ipynb`
    - Transformer:
        - `notebooks/transformers_final.ipynb`

---

### **STEP 2: Fine-tuning on MIMIC-IV Notes (Diagnosis Prediction)**

- Goal: Adapt pretrained models to specific clinical note understanding.
- Dataset: MIMIC-IV Notes  
- Task: **Multi-label** diagnosis classification (grouped ICD codes)
- Notebook References:
    - LSTM:
        - `notebooks/mimic_finetuning_pretrained_lstm.ipynb`
    - LSTM + Self-Attention: 
        - `notebooks/mimic_200_ctx_lstm_attention_demo.ipynb`


### **Additional Experiments: Medical Abstracts Dataset**

Before gaining access to the MIMIC-IV Notes dataset, we validated our pipeline on the Medical Abstracts dataset.

- **Task**: Multi-class classification
- Notebooks per Architecture:
    - LSTM:
        - From scratch: `notebooks/matc_lstm_demo.ipynb`
        - Fine-tuned: `notebooks/matc_lstm_fine_tuned_demo.ipynb`
    - LSTM + Self-Attention:
        - `notebooks/matc_lstm_attention_demo.ipynb`

---

## Get Started

### Clone the repository

```bash
git clone https://github.com/prashanthjaganathan/pretaining-language-models-for-medical-text.git
cd pertaining-language-models-for-medical-text
```

### Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 📥 Dataset Setup

> ⚠️ **Note:** Due to the large size of the datasets, we couldn’t host everything in a single downloadable link. You’ll need to download and set up the datasets manually on your machine. This means you’ll also need to run the **preprocessing and tokenization steps**.

We’ve tried to make things faster using the `pandarallel` library, but it can still take some time depending on your system. You can also check the output cells in the notebooks to understand what results to expect.

---

### 1. Setting Up the **MeDAL Dataset**

To download and prepare the MeDAL dataset:

1. Open the notebook  
   - `notebooks/medal_200_ctx_lstm_attention_demo.ipynb`

2. Run the **first two cells** in the notebook.  
   These cells will automatically download the MeDAL dataset and place it in the correct folder for the rest of the project to use.


### 2. Setting Up the **MIMIC-IV Notes Dataset**

To get started with the MIMIC-IV data:

1. Download the dataset from this Google Drive link:  
   🔗 [MIMIC-IV Drive Folder](https://drive.google.com/drive/folders/1oDAk8Mgc9o7JnFyJUm20LCS4sAmZizwC?usp=share_link)

2. Inside the downloaded folder, you'll find a directory called `MIMIC-IV`.

3. Move this entire `MIMIC-IV` folder into the project’s `dataset/` directory.  
   Your structure should look like this:
   ```
   project-root/
   └── dataset/
       └── MIMIC-IV/
           └── discharge.csv.gz
           └── diagosis.csv
           └── edstays.csv
   ```

4. Finally, unzip the `discharge.csv.gz` file:

---

## Load Trained Models
Load the necessary trained models from the google drive [link](https://drive.google.com/drive/folders/1_w_0wZB2kAnTcRxfyS7GE6mrJjVzELLx?usp=share_link). 

Download the `trained_models/` folder from the google drive and replace it with existing `trained_models/` folder in the project root. 

   Your structure should look like this:
   ```
   project-root/
   └── trained_models/
       └── models/
            └── medal_glove_100ctx_lstm_and_self_attention_model_model.pth
            └── ....
       └── embeddings/
       └── tokenizers/
   ```

---

## 🏃‍♂️ Running the Architectures

This project includes several notebooks you can run to train and evaluate different models on medical text datasets. Here's a clear breakdown by task and model:

### 1. Pretraining on MeDAL (Medical Abbreviation Disambiguation)

These notebooks train models to understand and disambiguate medical abbreviations from context.

- **🔹 LSTM Model:**  
  Run this notebook:  
  👉 `notebooks/medal_lstm_demo.ipynb`

- **🔹 LSTM + Self-Attention Model (200-token context):**  
  Run this notebook:  
  👉 `notebooks/medal_200_ctx_lstm_attention_demo.ipynb`

  We also experimented with shorter context windows:
  - 50 tokens: [`notebooks/medal_50_ctx_lstm_attention_demo.ipynb`](notebooks/medal_50_ctx_lstm_attention_demo.ipynb)
  - 100 tokens: [`notebooks/medal_100_ctx_lstm_attention_demo.ipynb`](notebooks/medal_100_ctx_lstm_attention_demo.ipynb)

- **🔹 Transformer Model:**
  Run this notebook to try out the Transformer architecture:  
  👉 `notebooks/transformers_final.ipynb`



### 🏥 2. Fine-tuning on MIMIC-IV Notes (Diagnosis Prediction)

These notebooks use the pretrained models from MeDAL and fine-tune them on the MIMIC-IV Notes dataset for multi-label diagnosis classification.

- **🔹 LSTM Model (Pretrained):**  
  👉 `notebooks/mimic_finetuning_pretrained_lstm.ipynb`

- **🔹 LSTM + Self-Attention Model (Pretrained, 200-token context):**  
  👉 `notebooks/mimic_200_ctx_lstm_attention_demo.ipynb`  
  ⚠️ *Note:* This notebook is still in progress — it currently runs into a `NaN` error during training.

---

### 3. Experiments with the MATC Dataset (Optional)

Before we had access to the full MIMIC-IV dataset, we tested our models on the [Medical Abstracts (MATC) dataset](https://huggingface.co/datasets/TimSchopf/medical_abstracts).

- **🔹 Fine-tuned LSTM Model:**  
  👉 `notebooks/matc_lstm_fine_tuned_demo.ipynb`

- **🔹 LSTM + Self-Attention Model (from scratch):**  
  👉 `notebooks/matc_lstm_attention_demo.ipynb`

---
Here’s a clean, concise, and user-friendly way to incorporate this flexibility into your GitHub README — right below the **Example: Generating Embeddings** section:

---

## Switching Between Tokenizers & Embedding Models

This repository is built with **modularity in mind**, allowing you to easily switch between different **tokenizers** and **embedding models** using the same code interface.

If you follow the documentation below, you'll be able to:

- **Choose from multiple tokenizers** using:
  ```python
  tokenized = dataset.tokenize('nltk', splits=['train'])
  ```
  Supported options:
  - `'whitespace'` – basic whitespace tokenizer
  - `'characters'` – character-level tokenizer
  - `'nltk'` – uses `nltk.word_tokenize`
  - `'pretrained'` – uses HuggingFace tokenizer (e.g., BioBERT, PubMedBERT)
  - `'trainable'` – learnable tokenizer like BPE or WordPiece

- **Switch between embedding types** like:
  ```python
  dataset.embed('bio_bert', ...)
  dataset.embed('bio_wordvec', ...)
  dataset.embed('trainable', ...)
  ```
  Supported embedding options:
  - `'bio_bert'` – for contextual BioBERT embeddings
  - `'bio_wordvec'` – pretrained static embeddings
  - `'trainable'` – embeddings learned from scratch (e.g., using FastText)

Just modify the `tokenizer_type` and `embedding_type` strings to plug in your preferred methods—no need to change anything else!

---

## Model Architectures

We evaluated three models end-to-end:

| Architecture | Pretraining | Fine-tuning | Embeddings |
|--------------|-------------|-------------|------------|
| **LSTM**     | ✅           | ✅           | GloVe       |
| **LSTM + Attention** | ✅     | Attempted         | GloVe       |
| **Transformer** | ✅        | X           | Glove and Bio+ClinicalBERT |

> Each model is implemented using PyTorch and trained for ~10–20 epochs depending on the stage.

---

## 📈 Performance Highlights

| Model                | Dataset     | Train Accuracy | Validation Accuracy |
|---------------------|-------------|----------------------|----------------|
| BiLSTM  | MeDAL       | 91.05%            | 83.73%         |
| BiLSTM + Attention  | MeDAL       | **92.60%**            | 84.40%         |
| Transformers  | MeDAL       | 74.76%           | 73.76%         |
| BiLSTM | MIMIC-IV    |             9.91%        | 5.07%              |
| BiLSTM + Attention  | MIMIC-IV    | 23.41%                  | 7.14%          |

---

## 📁 Repository Structure
```bash
/
├── README.md                     # Project overview, setup instructions, etc.
├── setup.py                      # Environment and sys.path setup
├── env.py                        # Defines project paths using Pathlib and Enum
├── classification_report.txt     # Output report from classification experiments
├── config/
│   └── config.yaml               # Configuration for datasets, models, embeddings, etc.
├── dataset/
│   ├── medal/                    # Pretraining dataset files
│   └── MIMIC-IV/                 # Fine-tuning dataset files (Notes)
|   └── MATC/                     # medical abstracts dataset
├── notebooks/
│   ├── transformers_final.ipynb  # Transformer experiments (pretraining, etc.)
│   ├── matc_lstm_fine_tuned_demo.ipynb      # Fine-tuning demo on Medical Abstracts (optional)
│   └── matc_lstm_attention_demo.ipynb         # LSTM+Attention demo on Medical Abstracts (optional)
│   └── ..... 
├── src/
│   └── data/
│       ├── base.py               # Base dataset class (ABCs, tokenization, embedding, etc.)
│       └── medal.py              # MeDALSubset class implementation
│       └── mimic.py              # MIMIC subset class implementation
│       └── matc.py               # MATC class implementation
│   └── models/
│   └── tokenizers/
│   └── vectorizer/
└── trained_models/
    ├── models/                   # Saved model checkpoints (e.g., PyTorch .pth files)
    ├── embeddings/               # Pretrained or generated embedding files
    └── tokenizers/               # Saved tokenizer models/configurations
```

---

## 🤝 Contributions

| **Team Member**       | **Contribution** |
|-----------------------|------------------|
| **Abhijit Kaluri**    | Built and trained a custom Transformer classifier using PyTorch with pre-trained GloVe embeddings, positional encoding, and a 2-layer Transformer encoder with 4 attention heads. |
| **Akshat Khare**      | Preprocessed, tokenized, and embedded the data, then trained a bidirectional LSTM model on both the MeDAL and MIMIC-IV datasets using PyTorch. |
| **Prashanth Jaganathan** | Preprocessed, tokenized, and embedded the data, then trained an LSTM + Self-Attention model on both the MeDAL and MIMIC-IV datasets using PyTorch. |

