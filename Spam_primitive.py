import statistics, time
import multiprocessing as mp
from pathlib import Path
from sklearn import svm, metrics
from dataset.Spam_dataset import SPAM_Parser
from dataset.methods import *
#This is the program to drow the PW noose to all attributions

to_mean = lambda l: f"{round(statistics.mean(l)*100, 2)}%"

def gen_data(_eps=-1, seed=0xE39FE39F, rand_label=False):
    dataset = SPAM_Parser("./dataset")

    if _eps != -1:
        eps = _eps / 34
        dataset.PW(eps, seed)
        if rand_label:
            dataset.random_label(eps, seed)
    
    return dataset

#def all_raw(c=2.1):
def all_raw():
    accs, recalls, precisions = [], [], []
    train_set = gen_data()
    test_set = gen_data()

    for i in range(10):
        data_train, label_train = train_set.split(i)
        #clf = svm.SVC(C = c)
        clf = svm.SVC(C = 4.1)
        clf.fit(data_train, label_train)

        data_test, label_test = test_set.split(i, True)
        pre = clf.predict(data_test)
        ac_score = metrics.accuracy_score(label_test, pre)
        recall = metrics.recall_score(label_test, pre)
        precision = metrics.precision_score(label_test, pre, zero_division=0)
        accs.append(ac_score)
        recalls.append(recall)
        precisions.append(precision)
    
    Path(f"result_Spam").mkdir(parents=True, exist_ok=True)
    with open(f"result_Spam/primitive.csv", 'a') as ofile:
        ofile.write(f"-1,{to_mean(accs)},{to_mean(recalls)},{to_mean(precisions)}\n")
        #ofile.write(f"-1,{to_mean(accs)},{to_mean(recalls)},{to_mean(precisions)},{c}\n")

def ten_seed(eps=3):
    accs, recalls, precisions = [], [], []

    seeds = [pow(0xE39FE39F, i, 1 << 32) for i in range(1, 11)]
    for seed in seeds:
        train_set = gen_data(eps, seed, True)
        test_set = gen_data(eps, seed)

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

def one_proc(begin, size):
    eps = begin
    while eps < begin + size +0.01:
        got = ten_seed(eps)
        Path(f"result_Spam").mkdir(parents=True, exist_ok=True)
        with open(f"result_Spam/primitive.csv", 'a') as ofile:
            ofile.write(f"{eps},{got[0]},{got[1]},{got[2]}\n")
        #eps +=0.2
        eps +=5
        eps = round(eps,2)

def all_batch(max_thread=20):
    procs = []
    #for eps in range(5, 50, 15):
    for eps in range(55, 100, 15):
        p = mp.Process(target=one_proc, args=(eps, 15))
        procs.append(p)
        p.start()

        if len(procs) == max_thread:
            wait_single(procs)
            procs = procs[1:]
    
    wait_all(procs)
    
def Grid_search(max_thread=20):
    procs = []
    C = [i for i in range(10,50,5)]
    #C = [i/10 for i in range(50)]
    for c in C:
        p = mp.Process(target=all_raw, args=(c, ))
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
    #all_raw()#ノイズを載せない場合の値
    #Grid_search()
    print(f"Time usage: {time.time() - start}")
