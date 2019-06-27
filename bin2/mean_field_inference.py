import numpy as np
from scipy.special import softmax


def one_hot_encoding(image, num_classes):
    h, w = image.shape
    one_hot_image = np.zeros((h, w, num_classes))
    for i in range(h):
        for j in range(w):
            _class = image[i, j] if image[i, j] < num_classes else 0
            one_hot_image[i, j, _class] = 1

    return one_hot_image


def get_neighbors(i, j, h, w):
    candidates = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    return [(a, b) for a, b in candidates if a >= 0 and b >= 0 and a < h and b < w]


def vector_transpose(vector):
    return vector.reshape(1, -1).T


def distance(a, b):
    return -np.linalg.norm(a - b)


def denoise_image(image, theta=0.5, threshold=0.01, num_colors=None):
    h, w = image.shape
    num_colors = np.max(image) + 1 if num_colors is None else num_colors
    one_hot_image = one_hot_encoding(image, num_colors)
    pi = np.zeros_like(one_hot_image)
    cmap = np.eye(num_colors)

    while True:
        pi_prev = np.copy(pi)

        for i in range(h):
            for j in range(w):
                scores = np.zeros(num_colors)
                for k in range(num_colors):
                    # theta Hi Hj term for all neighbors Hj
                    for x, y in get_neighbors(i, j, h, w):
                        scores[k] += theta * distance(pi[x, y], cmap[k])

                    # Hi Xj term
                    scores[k] += distance(one_hot_image[i, j], cmap[k])
                pi[i, j] = softmax(scores)

        delta = np.linalg.norm(pi - pi_prev)
        if delta < threshold:
            break

    denoised_image = np.argmax(pi, axis=2)

    return denoised_image
