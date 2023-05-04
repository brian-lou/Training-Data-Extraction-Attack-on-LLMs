import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

with open(args.results_path) as file:
    results = json.load(file)

scores_results = [score_result for score_result in zip(results['predictions']['real'], results['raw_results']) if score_result[1]['original_ll'] > -1000]
scores_samples = [(score, result['original']) for score, result in scores_results]
sorted_samples = [sample for _, sample in sorted(scores_samples)]

with open(args.output_path, 'w') as file:
    json.dump(sorted_samples, file)
