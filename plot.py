import matplotlib.pyplot as plt
import seaborn as sns
import json

def get_perp_scores(file_path):
    with open(file_path) as file:
        return json.load(file)

def get_detectgpt_scores(file_path):
    with open(file_path) as file:
        return json.load(file)['predictions']['real']

def plot_perp_detectgpt(perp_scores, detectgpt_scores, title, x_label, y_label, dest_path):
    sns.scatterplot(x=perp_scores, y=detectgpt_scores)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(dest_path)
    plt.close()

def plot_perp(perp_scores, title, x_label, y_label, dest_path):
    sns.histplot(x=perp_scores, bins=50)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(dest_path)
    plt.close()

sns.set_theme()

gpt2xl_perp = get_perp_scores('gpt-2-xl_perp-Perplexity_Scores.txt')
gpt2xl_detectgpt = get_detectgpt_scores('detect-gpt/final_results/gpt2-xl_t5-3b/perturbation_10_z_results.json')

gpt2imdb_perp = get_perp_scores('gpt-2-imdb_perp-Perplexity_Scores.txt')
gpt2imdb_detectgpt = get_detectgpt_scores('detect-gpt/final_results/gpt2-imdb_t5-3b/perturbation_10_z_results.json')

llama_perp = get_perp_scores('llama-samples-perp-Perplexity_Scores.txt')
llama_detectgpt = get_detectgpt_scores('detect-gpt/final_results/llama-_t5-3b/perturbation_10_z_results.json')

plot_perp_detectgpt(gpt2xl_perp, gpt2xl_detectgpt, 'GPT-2 XL', 'Perplexity score', 'DetectGPT score', 'plots/gpt2xl_perp_detectgpt.png')
plot_perp_detectgpt(gpt2imdb_perp, gpt2imdb_detectgpt, 'GPT-2 IMDB', 'Perplexity score', 'DetectGPT score', 'plots/gpt2imdb_perp_detectgpt.png')
plot_perp_detectgpt(llama_perp, llama_detectgpt, 'LLaMA', 'Perplexity score', 'DetectGPT score',  'plots/llama_perp_detectgpt.png')

plot_perp(gpt2xl_perp, 'GPT-2 XL', 'Perplexity score', 'Count', 'plots/gpt2xl_perp.png')
plot_perp(gpt2imdb_perp, 'GPT-2 IMDB', 'Perplexity score', 'Count', 'plots/gpt2imdb_perp.png')
plot_perp(llama_perp, 'LLaMA', 'Perplexity score', 'Count', 'plots/llama_perp.png')
