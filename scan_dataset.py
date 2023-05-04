import csv
import difflib
import json
import numpy as np

with open("IMDB Dataset.csv", mode='r') as f:
    # dataset = list(csv.reader(f, delimiter=","))
    dataset = ""
    csvreader = csv.reader(f)
    for row in csvreader:
        dataset += row[0]
    
def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines() 
    
    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs)-1):
        start = start_idxs[i]
        end = start_idxs[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            count_eng += 1
            for j in range(start+10, end):
                all_eng += lines[j]

    return all_eng

# dataset = parse_commoncrawl("commoncrawl.warc.wet")
def select_samples(ls, n=100):
    s = np.random.choice(ls, n, replace=False)
    return s

with open("gpt-2-imdb2.txt", mode = 'r', encoding="utf-8") as s:
    ls = json.load(s)
    samples = select_samples(ls)
    
# print(ls)
with open("gpt-2-xl.txt", 'r', encoding="utf-8") as s:
    ls = json.load(s)
    
with open("llama-samples-perp.txt", encoding="utf-8") as s:
    ls = json.load(s)
    # print(ls)

# Naive substring match
# for sample in samples:
#     if len(sample) <= 2:
#         continue
#     # for str in dataset:
#     #     if sample in str:
#     #         print((sample, str))
#     if sample in dataset:
#         print("Found")
            
# difflib
for sample in samples:
    if len(sample) <= 2:
        continue
    # print(difflib.get_close_matches(sample, dataset, n=3))
    match = difflib.SequenceMatcher(None, sample, dataset).find_longest_match()
    print("====================")
    print(sample[match.a:match.a + match.size])  
    print(dataset[match.b:match.b + match.size])