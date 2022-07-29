from sklearn import svm, metrics
from dataset.Spam_dataset import SPAM_Parser
import statistics
import multiprocessing as mp

def train_test(dataset):
    accs = []

    for i in range(10):
        data_train, label_train = dataset.split(i)
        clf = svm.SVC(C=3.9)
        clf.fit(data_train, label_train)
        
        data_test, label_test = dataset.split(i, True)
        pre = clf.predict(data_test)
        ac_score = metrics.accuracy_score(label_test, pre)
        accs.append(ac_score)

    return statistics.mean(accs)

def iterate_next(l, size, curr):
    if curr[0] == l[-size]:
        return []
    
    last_i = len(l) - 1
    for i in range(size-1, -1, -1):
        if curr[i] == last_i - (size - 1 - i):
            continue
        next_curr = curr[:i] + list(range(curr[i]+1, curr[i]+1 + size - i))
        break
    return next_curr

def test_columns(columns, queue):
    dataset = SPAM_Parser("./dataset")
    ori_data = dataset.data
    current_best = -1
    best_c = []
    for c in columns:
        dataset.data = ori_data[:, c]
        got = train_test(dataset)
        if got > current_best:
            current_best = got
            best_c = c
    queue.put((current_best, best_c))

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

def all_batch(dataset, size, max_thread=25):
    procs = []
    q = mp.Queue()
    curr = list(range(size))
    l = list(range(dataset.data.shape[1]))
    while True:
        columns = []
        while len(columns) < 500 and len(curr) > 0:
            columns.append(curr)
            curr = iterate_next(l, size, curr)
        p = mp.Process(target=test_columns, args=(columns, q))
        procs.append(p)
        p.start()

        if len(procs) == max_thread:
            print(curr)
            wait_single(procs)
            procs = procs[1:]
        
        if len(curr) == 0:
            wait_all(procs)
            break
    results = []
    while not q.empty():
        results.append(q.get())
    max_data = (-1, [])
    for r in results:
        if r[0] > max_data[0]:
            max_data = r
    return max_data

if __name__ == '__main__':
    dataset = SPAM_Parser("./dataset")
    print(all_batch(dataset, 3))
