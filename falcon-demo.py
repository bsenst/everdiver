# https://vilsonrodrigues.medium.com/run-your-private-llm-falcon-7b-instruct-with-less-than-6gb-of-gpu-using-4-bit-quantization-ff1d4ffbabcc
# https://github.com/vilsonrodrigues/playing-with-falcon/blob/main/notebooks/falcon-7b-instruct-4bit.ipynb

import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# My version with smaller chunks on safetensors for low RAM environments
model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        offload_folder="save_folder",
        )

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=296,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

pipeline("""Girafatron is obsessed with giraffes, the most glorious animal 
    on the face of this Earth. Giraftron believes all other animals 
    are irrelevant when compared to the glorious majesty of the 
    giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:""")