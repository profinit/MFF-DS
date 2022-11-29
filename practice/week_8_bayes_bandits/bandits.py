import numpy as np
import copy
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Bandit:

    rate = 0.0      # bandit return rate 0-1

    def __init__(self, return_rate = -1):
        self.rate = return_rate if return_rate >= 0 else np.random.random()

    def __str__(self):
        return f'B({self.rate:.3f})'

    def throw(self, n=1):
        return np.sum(np.random.random(n) <= self.rate)
    

class Bandits:

    n_bandits = 1           # number of bandits
    bandits = []            # list of bandits
    thrown = None           # coins thrown to bandits
    returned = None         # coins returned from bandits
    tot_thrown = 0          # total thrown coins
    tot_returned = 0        # total returned coins

    def reset(self):
        self.tot_thrown = 0          
        self.tot_returned = 0        
        self.returned = np.zeros(self.n_bandits, dtype=int)
        self.thrown = np.zeros(self.n_bandits, dtype=int)
    
    def __init__(self, n_bandits, rates = None, balanced = False):
        self.n_bandits = n_bandits
        if not rates: rates = np.random.random(n_bandits)
        if balanced: rates = n_bandits * rates / (2 * sum(rates))
        self.bandits = []
        for i in range(n_bandits):
            self.bandits.append(Bandit(rates[i]))
        self.reset()

    def throw(self, strategy, n=1):
        """ throws n coins into bandits according to strategy """
        for i in range(n):
            b = strategy(self.thrown, self.returned)
            r = self.bandits[b].throw(1)
            self.returned[b] += r
            self.thrown[b] += 1
            self.tot_returned += r
            self.tot_thrown += 1

        return self.tot_returned

    def __str__(self):
        return f'[{",".join(map(str,self.bandits))}]'


class Strategy(ABC):
    
    n_bandits = None
    
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits
    
    @abstractmethod
    def __call__(self, thrown, returned):
        pass

    @classmethod
    def name(cls):
        return cls.__name__
        
    
class RepeatStrategy(Strategy):
    
    last_bandit = 0
    last_return = 0
    
    def __init__(self, n_bandits):
        super().__init__(n_bandits)
        self.last_bandit = 0
        self.last_return = 0        
    
    def __call__(self, thrown, returned):
        
        if returned[self.last_bandit] <= self.last_return:
            self.last_bandit = np.random.randint(0, self.n_bandits, 1)[0]
    
        self.last_return = returned[self.last_bandit]
        return self.last_bandit
        
    
class BayesStrategy(Strategy):
    """ samples bandits randomly """
    
    def __init__(self, n_bandits, base = 2.0):
        super().__init__(n_bandits)
        self.base = base
    
    def __call__(self, thrown, returned):
        
        self.alphas = 0.5 * self.base + returned
        self.betas = 0.5 * self.base + thrown - returned
        
        bts = [np.random.beta(self.alphas[j], self.betas[j], 1)[0] for j in range(self.n_bandits)]
        return np.argmax(bts)
    