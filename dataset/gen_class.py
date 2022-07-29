import numpy as np
import time

def classification(data, num_classes):
    class_size = data.shape[1] // num_classes

    mids = []
    thresholds = []
    for i in range(data.shape[0]):
        sorted_rows = np.sort(data[i])
        sub_mids = []
        sub_thresholds = []
    
        for j in range(num_classes):
            sub_data = sorted_rows[j*class_size:(j+1)*class_size]
            sub_mids.append(np.median(sub_data))
            if j < num_classes - 1:
                sub_thresholds.append(sub_data[-1])

        mids.append(sub_mids)
        thresholds.append(sub_thresholds)
    
    return mids, thresholds

def get_data(dataset, eps=None, seed=None):
    dataset.deskew_all()
    dataset.projection()

    if eps is not None:
        dataset.add_noise(eps, seed)

    data = np.array(dataset.imgs).T
    return data


def print_result(mids, thresholds):
    print("Medians")
    for i in range(mids.shape[0]):
        print(i, mids[i])

    print("Thresholds")
    for i in range(thresholds.shape[0]):
        print(i, thresholds[i])


def gen_WA(data, num_classes=5, do_print=False):
    mids, thresholds = map(np.array, classification(data.T, num_classes))

    if do_print:
        print_result(mids, thresholds)

    return mids, thresholds


def gen_DP_WA(eps=10, id=1, do_print=False):
    data = get_data(eps, pow(0xE39FE39F, id, 1 << 32))
    mids, thresholds = map(np.array, classification(data, 10))

    if do_print:
        print_result(mids, thresholds)

    to_str = lambda num: str(num).replace(".", "_")
    np.save(f"dataset/classes/eps{to_str(eps)}-medians-{id}", mids)
    np.savetxt(f"dataset/classes-csv/eps{to_str(eps)}-medians-{id}.csv", mids, delimiter=",")

    np.save(f"dataset/classes/eps{to_str(eps)}-thresholds-{id}", thresholds)
    np.savetxt(f"dataset/classes-csv/eps{to_str(eps)}-thresholds-{id}.csv", thresholds, delimiter=",")


def gen_fixed(tot_attr, n_class=5):
    thresholds = np.array([[-1 + i * (2 / n_class) for i in range(1, n_class)] for _ in range(tot_attr)])
    mids = np.array([[-1 + (i + 0.5) * (2 / n_class) for i in range(n_class)] for _ in range(tot_attr)])
    return mids, thresholds

if __name__ == '__main__':
    print("Begin:", time.asctime())

    print(gen_fixed(1, 4))
