import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

with open('gpt-2-xl_perp-Perplexity_Scores.txt') as file:
    perp_scores = json.load(file)

with open('detect-gpt/final_results/gpt2-xl_t5-3b/perturbation_10_z_results.json') as file:
    detectgpt_scores = json.load(file)['predictions']['real']

def plot_perp_detectgpt(perp_scores, detectgpt_scores, dest_path):
    sns.scatterplot(x=perp_scores, y=detectgpt_scores)
    plt.savefig(dest_path)

gpt2xl_perp = 'gpt-2-xl_perp-Perplexity_Scores.txt'
gpt2xl_detectgpt = 'detect-gpt/final_results/gpt2-xl_t5-3b/perturbation_10_z_results.json'
gpt2imdb_perp = 'gpt-2-xl_perp-Perplexity_Scores.txt'
gpt2imdb_detectgpt = 'detect-gpt/final_results/gpt2-xl_t5-3b/perturbation_10_z_results.json'
llama_perp = 'gpt-2-xl_perp-Perplexity_Scores.txt'
llama_detectgpt = 'detect-gpt/final_results/gpt2-xl_t5-3b/perturbation_10_z_results.json'

