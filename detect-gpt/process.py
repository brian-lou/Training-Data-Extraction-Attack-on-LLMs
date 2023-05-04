import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

with open(args.results_path) as file:
    results = json.load(file)

scores = results['predictions']['real']
samples = [result['original'] for result in results['raw_results']]
scores_samples = sorted(zip(scores, samples))
sorted_samples = [sample for _, sample in scores_samples]

with open(args.output_path, 'w') as file:
    json.dump(sorted_samples, file)
