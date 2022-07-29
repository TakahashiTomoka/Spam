import math
import numpy as np

def PM_batch(data, rng, eps=3):
    eps_exp = math.exp(eps/2)
    C = (eps_exp + 1) / (eps_exp - 1)
    L = data * ((C + 1) / 2) - ((C - 1) / 2)
    R = L + C - 1

    X = rng.uniform(0, 1, data.shape)

    uniform_LR = rng.uniform(L, R)
    # C - r + l + C = 2C + 1 - C = C + 1
    uniform_out = rng.uniform(-C, 1, data.shape)
    uniform_out = np.where(uniform_out >= L, uniform_out + C - 1, uniform_out)

    return np.where(X >= eps_exp / (eps_exp + 1), uniform_out, uniform_LR)

if __name__ == '__main__':
    pass
