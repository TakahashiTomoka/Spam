import statistics, time
import multiprocessing as mp
from pathlib import Path
from sklearn import svm, metrics
from dataset.Spam_dataset import SPAM_Parser
from dataset.methods import *

to_mean = lambda l: f"{round(statistics.mean(l)*100, 2)}%"

def gen_data(attr_size=3, attr=NO_NOISE, _type=NO_NOISE, _eps=3, seed=0xE39FE39F, random_label=False, n_class=5):
    eps = _eps / (2 * (attr_size+1)) if attr in [PW, WADP] else _eps / (attr_size+1)
    dataset = SPAM_Parser("./dataset")
    #step1 = attr
    #属性を決定するフェーズ
    if attr == PW:
        dataset.reduce_attributes(attr_size, True, PW, eps, seed)
    elif attr == WADP:
        dataset.reduce_attributes(attr_size, True, WADP, eps, seed, n_class)
    elif attr == RANDOM:
        dataset.reduce_attributes(attr_size, True, RANDOM)
    elif attr == WA:
        dataset.reduce_attributes(attr_size, False, WA)
    else:
        dataset.reduce_attributes(attr_size)
    
    #train or testに応じてノイズをかけていくフェーズ
    if _type == WA:
        dataset.WA()
    elif _type == PW:
        dataset.PW(eps, seed)
    elif _type == WADP:
        dataset.WADP(eps, seed, n_class)
    
    #trainingデータに対しては、目的変数に対してノイズをのせる
    if random_label:
        dataset.random_label(eps, seed)
    
    return dataset

def ten_seed(attr_size=3, attr=NO_NOISE, eps=3, train_type=NO_NOISE, test_type=NO_NOISE, n_class=5):
    accs, recalls, precisions = [], [], []

    seeds = [pow(0xE39FE39F, i, 1 << 32) for i in range(1, 11)]
    for seed in seeds:#属性を決定
        train_set = gen_data(attr_size, attr, train_type, eps, seed, True, n_class)
        test_set = gen_data(attr_size, attr, test_type, eps, seed, False, n_class)

        for i in range(10):#決定した属性に対して交差検定
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
    
    return to_mean(accs), to_mean(recalls), to_mean(precisions)

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

def one_proc(step1=RANDOM, train_type=NO_NOISE, test_type=NO_NOISE, attr_size=3, n_class=5, eps_begin=3, eps_end=5):
    name = f"step1={names[step1]} train={names[train_type]} test={names[test_type]} attr={attr_size} class={n_class}"
    print(f"{name} start")

    eps = eps_begin
    while eps <= eps_end+0.05:
        got = ten_seed(attr_size, step1, eps, train_type, test_type, n_class)
        Path(f"result_Spam/{names[step1]}/{names[train_type]}_{names[test_type]}").mkdir(parents=True, exist_ok=True)
        with open(f"result_Spam/{names[step1]}/{names[train_type]}_{names[test_type]}/{attr_size}_C{n_class}.csv", 'a') as ofile:
            ofile.write(f"{round(eps, 2)},{attr_size},{n_class},{names[step1]},{got[0]},{got[1]},{got[2]}\n")
        eps += 0.2
        eps = round(eps, 2)
    
    print(f"{name} done")

def all_batch(max_thread=20):
    procs = []
    for attr_size, n_class in [(6,2),(7,4)]:
        for eps in range(5, 50, 15):
            p = mp.Process(target=one_proc, args=(RANDOM, WA, WADP, attr_size, n_class, eps, eps+15))
            procs.append(p)
            p.start()
            if len(procs) == max_thread:
                wait_single(procs)
                procs = procs[1:]    
    wait_all(procs)

def all_batch2(max_thread=20):
    for step1 in [WA, RANDOM, PW]:
        for attr_size in [5,6,7,8,9,10]:
            procs = []
            for n_class in [2,3,4,5]:#(2, 3),(4,4),(6,5)
                for eps in range(5, 50, 15):
                    p = mp.Process(target=one_proc, args=(step1, WADP, WADP, attr_size, n_class, eps, eps+15))
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