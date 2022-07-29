import math
import numpy as np

# min is always 0
def discrete_noise(data, rng, eps=3, max_val=9):
    eps_exp = math.exp(eps)
    other_prob = 1 / (max_val + eps_exp)
    threshold = eps_exp * other_prob

    sample = rng.uniform(0, 1, data.shape)#[0,1)でdata.shapeと同じなさの配列を生成
    new_val = np.where(sample >= (1 - threshold), max_val, sample // other_prob)
    new_val = np.where(new_val >= data, new_val + 1, new_val)
    new_val = np.where(new_val == max_val + 1, data, new_val)
    return new_val.astype(int)


if __name__ == '__main__':
    eps = 2
    data = [[0, 1, 2], [3, 4, 5]]
    rng = np.random.default_rng(0xE39FE39F)
    got = discrete_noise(np.array(data), rng, eps)
    print(got)
    