import torch

def read_data(filename):

    sentences = []

    with open(filename) as file:
        for line in file:
            temp = line[0] + ' '
            temp += line[1:].strip()
            sentences.append(temp)
    return sentences


def subset_idxs_dist(dataset, classes, set_size, balanced=True, randomized=False):
    from numpy import random
    
    data_idxs =list(i for i in range(len(dataset)))
    set_dist = {c: 0 for c in classes}
    set_idxs = []
    
    idx = 0

    for i in range(set_size):
        if randomized == True:
            idx = data_idxs[random.randint(len(data_idxs))]

        while int(dataset[idx][:2]) not in classes:
            data_idxs.remove(idx)
            if randomized == False: idx += 1
            else: idx = data_idxs[random.randint(len(data_idxs))]

        if balanced == True:

            while int(dataset[idx][:2]) not in classes or set_dist[int(dataset[idx][:2])] >= set_size//len(classes):
                data_idxs.remove(idx)
                if randomized == False: idx += 1
                else: idx = data_idxs[random.randint(len(data_idxs))]

        set_dist[int(dataset[idx][:2])] += 1
        set_idxs.append(idx)
        data_idxs.remove(idx)
        if randomized == False: idx += 1

    return set_idxs, set_dist

def words_to_idxs(dataset, idxs):

    words_to_idxs = {}

    for idx in idxs:
        for word in dataset[idx][2:-1].split():
            if word not in words_to_idxs: words_to_idxs[word] = len(words_to_idxs)

    return words_to_idxs

def prepare_sequence(sequence, to_idx):
    try:
        idxs = [to_idx[word] for word in sequence]
        return torch.tensor(idxs, dtype=torch.long)
    except:
        idxs = [word for word in sequence]
        return torch.tensor(idxs, dtype=torch.long)
