import numpy as np

# data shape: [total records, total attributes]
# mids shape: [total attributes, number of classes]
def to_classes(data, mids):
    return np.choose(data, mids.T)

# data shape: [total records, total attributes]
# thresholds: [total attributes, number of classes - 1]
def to_index(data, thresholds):
    output = []
    tot_attr = data.shape[1]
    for attr in range(tot_attr):
        new_c = np.searchsorted(thresholds[attr], data.T[attr])
        output.append(new_c)
    return np.array(output).T
