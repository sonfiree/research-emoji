import tensorflow as tf
import random

class BatchNegSampler:
    def __init__(self, phr_to_ind, emoj_to_ind, batch_size, neg_per_pos, train_set_true, train_set_false, dev_set=None):
        self.batch_size = batch_size
        self.neg_per_pos = neg_per_pos
        self.train_set_true = train_set_true
        self.train_set_false = train_set_false
        self.dev_set = dev_set
        self.epoch = 0
        self.phr_to_ind = phr_to_ind
        self.emoj_to_ind = emoj_to_ind
        self.iter_pos = 0
        self.iter_neg = 0
        random.shuffle(self.train_set_true)
        random.shuffle(self.train_set_false)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_pos >= len(self.train_set_true):
            random.shuffle(self.train_set_true)
            self.iter_pos = 0
            
        if self.iter_neg >= len(self.train_set_false):
            random.shuffle(self.train_set_false)
            self.iter_neg = 0
            self.epoch += 1
            raise StopIteration
        return self.get_batch(neg_per_pos=self.neg_per_pos)

    # @profile
    def get_batch(self, neg_per_pos=1):
        num_pos = self.batch_size // (1 + neg_per_pos)
        num_neg = self.batch_size - num_pos
        pos = []
        neg = []
        for i in range(num_pos):
            pos.append(self.train_set_true[self.iter_pos%len(self.train_set_true)])
            self.iter_pos += 1
        for i in range(num_neg):
            neg.append(self.train_set_false[self.iter_neg%len(self.train_set_false)])
            self.iter_neg += 1
            
        batch = pos + neg
        
        rows = list()
        cols = list()
        targets = list()
        
        for i in range(len(batch)):
            example = batch[i]
            rows.append(self.phr_to_ind[example[0]])
            cols.append(self.emoj_to_ind[example[1]])
            if example[2] == 'True':
                targets.append(1)
            else:
                targets.append(0)
        return cols, rows, targets
