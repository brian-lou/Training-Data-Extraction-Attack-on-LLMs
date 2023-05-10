import json 


import logging
logging.basicConfig(level='ERROR')

import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import os
import utils
os.environ['TRANSFORMERS_CACHE'] = '/scratch/gpfs/blou/.cache/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="/scratch/gpfs/blou/.cache/")
# tokenizer = LlamaTokenizer.from_pretrained("/scratch/gpfs/blou/.llamahugging")
tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb", cache_dir="/scratch/gpfs/blou/.cache/", padding_side = "left" )
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token

# model1 = LlamaForCausalLM.from_pretrained("/scratch/gpfs/blou/.llamahugging").cuda()
model1 = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb", return_dict=True, cache_dir="/scratch/gpfs/blou/.cache/").to(device)
# model1 = GPT2LMHeadModel.from_pretrained('gpt2-xl', return_dict=True, cache_dir="/scratch/gpfs/blou/.cache/").to(device)
model1.config.pad_token_id = model1.config.eos_token_id
model1.eval()
    
# filename = "gpt-2-xl_perp.txt"
# savename = "gpt-2-xl_perp-Perplexity_Scores.txt"
filename = "gpt-2-imdb_perp.txt"
savename = "gpt-2-imdb_perp-Perplexity_Scores.txt"
# filename = "llama-samples-perp.txt"
# savename = "llama-samples-perp-Perplexity_Scores.txt"
with open(filename, encoding="utf-8", mode="r") as s:
    ls = json.load(s)
    
perp = []
for sample in tqdm(ls, desc="Calculating:"):
    perp.append(calculatePerplexity(sample, model1, tokenizer).to('cpu').item())

with open(savename, encoding="utf-8", mode="w") as s:
    s.write(json.dumps(perp))