"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""

import csv
import json
import logging
logging.basicConfig(level='ERROR')

import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
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
    # print(f"Sentence: {sentence}")
    tmp = tokenizer.encode(sentence)
    # print(f"Tokens: {tmp}")
    input_ids = torch.tensor(tmp).unsqueeze(0)
    # print(input_ids)
    input_ids = input_ids.to(device)
    # print(input_ids)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

model_name = "lvwerra/gpt2-imdb"
cache_dir = "/scratch/gpfs/blou/.cache/"

if "llama" in model_name:
    tokenizer = LlamaTokenizer.from_pretrained("/scratch/gpfs/blou/.llamahugging")
    model1 = LlamaForCausalLM.from_pretrained("/scratch/gpfs/blou/.llamahugging").cuda()
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side = "left" )
    model1 = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, cache_dir=cache_dir).to(device)

tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token

# model1 = GPT2LMHeadModel.from_pretrained('gpt2-GPT2-IMDB', return_dict=True, cache_dir="/scratch/gpfs/blou/.cache/").to(device)

model1.config.pad_token_id = model1.config.eos_token_id
# model2 = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True, cache_dir="/scratch/gpfs/blou/.cache/").to(device)
model1.eval()
    
with open("gpt-2-imdb2.txt", mode = 'r', encoding="utf-8") as s:
    ls = json.load(s)

out = []
for sent in ls:
    p1 = calculatePerplexity(ls, model1, tokenizer)
    out.append({"sample":sent, "ppl":p1})

with open("gpt-2-imdb_perp.txt", mode = 'w+', encoding="utf-8") as file:
    file.write(json.dumps(out))