import numpy as np
from tqdm import tqdm

from utls.utls import l2_norm


def calculator_distance(embeddings):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), axis=1)

    return dist


def evalute(embedding_size, batch_size, model, carray, issame):
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
        batch = batch[:, :, :, ::-1]

        embeding_batch = model(batch)
        embeddings[idx:idx + batch_size] = l2_norm(embeding_batch)

    distance_batches = calculator_distance(embeddings)

    distances = np.array(distance_batches)
    labels = np.array(issame)
    return distances, labels
