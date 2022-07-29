import statistics, time
import multiprocessing as mp
from pathlib import Path
from sklearn import svm, metrics
from dataset.Spam_dataset import SPAM_Parser
from dataset.methods import *
import warnings
warnings.filterwarnings('ignore')

to_mean = lambda l: f"{round(statistics.mean(l)*100, 2)}%"

def gen_data(attr_size=3, attr=NO_NOISE, _type=NO_NOISE, _eps=3, seed=0xE39FE39F, random_label=False, n_class=5):
    eps = _eps / (2 * (attr_size+1)) if attr in [PW, WADP] else _eps / (attr_size+1)
    dataset = SPAM_Parser("./dataset")
    if attr == PW:
        dataset.reduce_attributes(attr_size, True, PW, eps, seed)
    elif attr == WADP:
        dataset.reduce_attributes(attr_size, True, WADP, eps, seed, n_class)
    elif attr == RANDOM:
        dataset.reduce_attributes(attr_size, False, RANDOM)
    elif attr == WA:
        dataset.reduce_attributes(attr_size, False, WA)
    else:
        dataset.reduce_attributes(attr_size)
    
    if _type == WA:
        dataset.WA()
    elif _type == PW:
        dataset.PW(eps, seed)
    elif _type == WADP:
        dataset.WADP(eps, seed, n_class)
    
    if random_label:
        dataset.random_label(eps, seed)
    
    return dataset

def ten_seed(attr_size=3, attr=NO_NOISE, eps=3, train_type=NO_NOISE, test_type=NO_NOISE, n_class=5):
    accs, recalls, precisions = [], [], []

    seeds = [pow(0xE39FE39F, i, 1 << 32) for i in range(1, 11)]
    for seed in seeds:
        train_set = gen_data(attr_size, attr, train_type, eps, seed, train_type in [WADP, PW], n_class)
        test_set = gen_data(attr_size, attr, test_type, eps, seed, False, n_class)

        for i in range(10):
            data_train, label_train = train_set.split(i)
            clf = svm.SVC(C=4.1)
            clf.fit(data_train, label_train)

            data_test, label_test = test_set.split(i, True)
            pre = clf.predict(data_test)
            ac_score = metrics.accuracy_score(label_test, pre)
            recall = metrics.recall_score(label_test, pre)
            precision = metrics.precision_score(label_test, pre, zero_division=0)
            accs.append(ac_score)
            recalls.append(recall)
            precisions.append(precision)
    
    return statistics.mean(accs), to_mean(accs)

def wait_single(procs):
    try:
        procs[0].join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        exit()

def wait_all(procs):
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        exit()

def one_proc(eps):
    print(f"{eps} start")
    best_acc = 0
    best_type = ""
    
    for attr in [WA]:#属性の決定方法
        for attr_size in [5,10,15,20]:
            for n_class in range(2, 5):
                for train_type in [WADP]:
                    for test_type in [WADP]:
                        got = ten_seed(attr_size, attr, eps, train_type, test_type, n_class)
                        if got[0] > best_acc:
                            best_acc = got[0]
                            best_type = f"{names[attr]},{attr_size},{n_class},{names[train_type]},{names[test_type]},{got[1]}"

    Path(f"result_Spam/{names[attr]}/{names[train_type]}_{names[test_type]}/").mkdir(parents=True, exist_ok=True)
    with open(f"result_Spam/{names[attr]}/{names[train_type]}_{names[test_type]}/combination_all.csv", 'a') as ofile:
        ofile.write(f"{eps},{best_type}\n")
    
    print(f"{eps} {best_acc} {best_type}")

def all_batch(max_thread=20):
    procs = []
    epses = [(10 + 5*i) for i in range(9)]
    for eps in epses:
        p = mp.Process(target=one_proc, args=(eps,))
        procs.append(p)
        p.start()

        if len(procs) == max_thread:
            wait_single(procs)
            procs = procs[1:]
    
    wait_all(procs)

if __name__ == '__main__':
    print("Begin:", time.asctime())
    start = time.time()
    all_batch()
    print(f"Time usage: {time.time() - start}")
