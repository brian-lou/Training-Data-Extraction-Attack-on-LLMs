import requests
import os
import ast
import numpy as np

API_KEY = os.environ.get("API_KEY")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
API_KEY = "AIzaSyDTWD9wDMkNqcNVmDYhe2opu3gLC0ImG0s"
SEARCH_ENGINE_ID = "a614f84609947433d"


def inverse_cdf(u):
    return int(1000 * (u**2))

def sample_elements(arr, num_samples):
    sampled_indices = set()
    while len(sampled_indices) < num_samples:
        u = np.random.uniform(0, 1)
        idx = inverse_cdf(u)
        if idx < len(arr):
            sampled_indices.add(idx)
    sampled_elements = [arr[i] for i in sorted(sampled_indices)]
    return sampled_elements

# print(sample_elements(sorted(np.random.rand(1000)), 100))



# Read search terms from file
with open("gpt-2-xl_perp.txt", "r") as infile:
    file_content = infile.read()
    SEARCH_TERMS = ast.literal_eval(file_content)

total_search_terms = len(SEARCH_TERMS)
successful_searches = 0



# Open a text file named "results.txt" for saving search results
with open("results_gpt-2-xl-all_1000_perp.txt", "w", encoding="utf-8", errors="surrogateescape") as results_file:
    for query in SEARCH_TERMS:
        url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q=%22{query}%22"
        response = requests.get(url)
        results = response.json()

        results_file.write("=============================================\n")

        if results.get('items'):
            results_file.write(f"Results found for: '{query}'\n")
            successful_searches += 1
            for item in results['items']:
                results_file.write(f"Title: {item['title']}\n")
                results_file.write(f"Link: {item['link']}\n\n")
        else:
            results_file.write(f"No results found for: '{query}'\n\n")

    results_file.write(f"Total search terms: {total_search_terms}\n")
    results_file.write(f"Total successful searches: {successful_searches}\n")
    results_file.write(f"Percentage of successful searches: {100 * successful_searches / total_search_terms}%\n")

print(f"Total search terms: {total_search_terms}")
print(f"Total successful searches: {successful_searches}")
print(f"Percentage of successful searches: {100 * successful_searches / total_search_terms}%")