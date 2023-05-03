import numpy as np
import json
def print_best_tofile(metric, samples, name1, scores1, file, name2=None, scores2=None, n=100):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]
    ls = []
    for i, idx in enumerate(idxs):
        if scores2 is not None:
            # print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
            pass
        else:
            # print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")
            pass
        ls.append(samples[idx])
    file.write(json.dumps(ls))