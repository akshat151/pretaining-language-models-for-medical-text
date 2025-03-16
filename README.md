# pertaining-language-models-for-medical-text
This repository focuses on pretraining language models on the MeDAL dataset for medical text, utilizing multiple architectures to enhance performance for medical NLP tasks.

Paper [Link](https://arxiv.org/pdf/2012.13978)

Here, we will be focusing on pretraining the MeDAL dataset using an LSTM + Self-Attention architecture.

## Getting Started
1. Clone the repository

2. To use the scispacy `en_core_sci_sm` model for lemmantizing as it is trained on biomedical data with the ~100k vocabulary. Run

```bash
python -m spacy download en_core_web_sm

```

3. 