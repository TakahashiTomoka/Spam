import os, csv, random
import numpy as np
from noise.LDP_piecewise import PM_batch
from noise.LDP_discrete import discrete_noise
from dataset.classify import to_classes, to_index
from dataset.gen_class import gen_fixed, gen_WA
from dataset.methods import *

class SPAM_Parser:    
    def __init__(self, path="."):
        #spam = r"spambase.csv"
        spam = r"C:\Users\vml97\OneDrive - Osaka University\ドキュメント\Osaka University\2nd Term\安全なデータ設計特論\Joey\Spam\dataset\spambase.csv"
        with open(spam) as infile:
            self.data = np.array(list(csv.reader(infile)), dtype=np.float)

        np.random.seed(0xE39FE39F)
        np.random.shuffle(self.data)

        # Split data and label
        self.data, self.label = self.data[:,1:-1], self.data[:,-1]
        
        # first normalization
        mean, std = np.mean(self.data, axis=0), np.std(self.data, axis=0)
        std = np.where(std == 0, 1, std)
        self.data = (self.data - mean) / std

        self.size = self.data.shape[0]
        self.filter = list(range(self.data.shape[1]))
        self.discrete_count = 0
    
    def reduce_attributes(self, size, randomly=False, noise=NO_NOISE, eps=3, seed=0xE39FE39F, n_class=5):
        tot_attr = self.data.shape[1]
        assert size <= tot_attr
        new_data, new_label = self.data, self.label

        if noise == WA:
            mids, thresholds = gen_fixed(new_data.shape[1], n_class)
            new_data = to_classes(to_index(new_data, thresholds), mids)
            test_label = np.where(self.label == 0, -1, self.label)
            test_data = np.copy(new_data)
            test_data[:,0] = np.where(test_data[:,0] == 0, -1, 1)
            got = [(abs(np.sum(test_data[:, i]*test_label)), i) for i in range(test_data.shape[1])]
            got = sorted(got, key=lambda c: c[0], reverse=True)
            self.filter = sorted([got[i][1] for i in range(size)])
            self.data = self.data[:, self.filter]
            return

        if noise == RANDOM:
            random.seed(seed)
            chosen = sorted(random.choices(list(range(tot_attr)), k=size))
            self.data = self.data[:, chosen]
            self.filter = chosen
            return

        if randomly:
            rng = np.random.default_rng(seed)

            r_data = np.copy(self.data)
            disc_data, cont_data = r_data[:,:self.discrete_count], r_data[:, self.discrete_count:]
            disc_data = discrete_noise(disc_data, rng, eps, 1)
            r_label = np.copy(self.label)
            r_label = discrete_noise(r_label, rng, eps, 1)

            if noise == WADP:
                mids, thresholds = gen_fixed(cont_data.shape[1], n_class)
                cont_data = to_index(cont_data, thresholds)
                cont_data = discrete_noise(cont_data, rng, eps, thresholds.shape[1])
                cont_data = to_classes(cont_data, mids)
            elif noise == PW:
                cont_data = PM_batch(cont_data, rng, eps)

            r_data = np.concatenate((disc_data, cont_data), axis=1)
            new_data = [[] for _ in range(tot_attr)]
            new_label = [[] for _ in range(tot_attr)]
            random.seed(seed)
            for i in range(self.size):
                picked = random.sample(range(tot_attr), size)
                for c in picked:
                    new_data[c].append(r_data[i][c])
                    new_label[c].append(r_label[i])

        if noise == WA:
            mids, thresholds = gen_fixed(new_data.shape[1], n_class)
            new_data = to_classes(to_index(new_data, thresholds), mids)

        corr = []
        for i in range(tot_attr):
            if randomly:
                coff = abs(np.corrcoef(new_data[i], new_label[i])[0][1])
            else:
                coff = abs(np.corrcoef(new_data[:,i], new_label)[0][1])
            corr.append((coff, i))
        corr = sorted(corr, key=lambda c: c[0], reverse=True)

        self.filter = sorted([corr[i][1] for i in range(size)])
        self.data = self.data[:, self.filter]

    def split(self, idx = 0, testing = False):
        assert idx < 10
        test_size = self.size // 10
        data_train = np.concatenate((self.data[0:idx*test_size], self.data[(idx+1)*test_size:]))
        mean, std = np.mean(data_train, axis=0), np.std(data_train, axis=0)
        std = np.where(std == 0, 1, std)
        if testing:
            data = self.data[idx*test_size:(idx+1)*test_size]
            label = self.label[idx*test_size:(idx+1)*test_size]
            return (data - mean) / std, label
        else:
            label = np.concatenate((self.label[0:idx*test_size], self.label[(idx+1)*test_size:]))
            return (data_train - mean) / std, label

    def WA(self):        
        if 0 in self.filter:
            disc_data, cont_data = self.data[:, :self.discrete_count], self.data[:, self.discrete_count:]
        else:
            cont_data = self.data
        mids, thresholds = gen_WA(cont_data)
        self.data = to_classes(to_index(cont_data, thresholds), mids)
        if 0 in self.filter:
            self.data = np.concatenate((disc_data, self.data), axis=1)
    
    def PW(self, eps=3, seed=0xE39FE39F):
        rng = np.random.default_rng(seed)
        if 0 not in self.filter:
            self.data = PM_batch(self.data, rng, eps)
        else:
            disc_data = discrete_noise(self.data[:,:self.discrete_count], rng, eps, 1)
            cont_data = PM_batch(self.data[:, self.discrete_count:], rng, eps)
            self.data = np.concatenate((disc_data, cont_data), axis=1)
    
    def WADP(self, eps=3, seed=0xE39FE39F, n_class=5):
        rng = np.random.default_rng(seed)
        if 0 in self.filter:
            disc_data, cont_data = self.data[:, :self.discrete_count], self.data[:, self.discrete_count:]
        else:
            cont_data = self.data
        mids, thresholds = gen_fixed(cont_data.shape[1], n_class)
        cont_data = to_index(cont_data, thresholds)
        cont_data = discrete_noise(cont_data, rng, eps, thresholds.shape[1])
        cont_data = to_classes(cont_data, mids)
        if 0 in self.filter:
            disc_data = discrete_noise(disc_data, rng, eps, 1)
            self.data = np.concatenate((disc_data, cont_data), axis=1)
        else:
            self.data = cont_data

    def random_label(self, eps=3, seed=0xE39FE39F):
        rng = np.random.default_rng(seed)
        self.label = discrete_noise(self.label, rng, eps, 1)