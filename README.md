# CBIR-Text---AI-Studio

## About
CBIR Text was developed by a team of 5: Didi Adeshina, Dechen Bhuming, Erica Egleston, Megan Lai, and Yuri Lee. Proposed and guided by Novartis industry professionals, this project aims to facilitate drug research and development by producing a more efficient way to query histopathology images. This model would allow pathologists to analyze and diagnose pathology images of the side effects of a drug candidate. This project was presented to Novartis pathologists and machine learning engineers to share the findings. 

The goal of this project is to find a way to query semantically similar captions with a given text query and to pull the top n relevant examples from the dataset. The approach to this was to preprocess the captions into word embeddings to capture the semantic meaning into a vector, to calculate the similarity between a query and the captions using an evalutation metric (Cosine Similarity).

This project was completed in various stages:
1. Understanding the example dataset
2. Researching NLP models relevant to the medical domain
3. Creating Word Embeddings
4. Finding results
5. Evaluating the results of the different models used
6. Automatically generate captions diagnosing an image

## Data Pre-processing

## Models
### PLIP (Pathology Language and Image Pre-training)
### PubMedBERT
### BioWordVec



## Evaluations

## Setup:
1. clone the PLIP repository and place this file in the top directory (plip):  
```
git clone https://github.com/PathologyFoundation/plip.git
```

2. download BioWordVec bin file (13GB) from https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin

3. Recommended to create an environment :
```
// (in bash)
python3 -m venv .topn
source .topn/Scripts/activate
```

5. Install libraries: numpy, torch, pillow, sentence-transformers, gensim, datasets


## Caption Generation with LLaVa-Med

Steps to generate captions:
1. Clone LLaVA-Med repo and cd into it:
```
git clone https://github.com/microsoft/LLaVA-Med.git
cd LLavA-Med
```
2. Download files from https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/tree/main into LLaVA-Med directory.
3. In llava/model/builder.py set the default device param to "cpu": device="cpu"
4. In builder.py, modify setting the tokenizer (line 30) to:

```
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
```

5. Load the model by running this in a code cell

```
from llava.model.builder import load_pretrained_model

# Load the model from the downloaded path
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path='./',  # Path where you cloned the repo
    model_base=None,
    model_name='llava-med-v1.5-mistral-7b',  # Model name,
    load_8bit=False, load_4bit=False, device_map="auto"
)
``` 
