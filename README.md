# CBIR-Text---AI-Studio

We are using the PLIP model for word embeddings and the LLaVA-Med model for generating captions for pathology images.

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
