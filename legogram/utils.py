from itertools import islice
from rdkit import Chem
from tqdm import tqdm, tqdm_notebook
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str: # jupyter
        tqdm = tqdm_notebook
    if 'terminal' in ipy_str: # ipython
        pass
except: pass # terminal

import numpy as np
import time
import multiprocessing as mp
import joblib

from collections import Counter, OrderedDict
from itertools import combinations 

import copy

def map_parallel(lst, fn, nworkers=1, fallback_sharing=False):
    if nworkers == 1:
        return [fn(item) for item in lst]

    if fallback_sharing:
        mp.set_sharing_strategy('file_system')
    L = mp.Lock()
    QO = mp.Queue(nworkers)
    QI = mp.Queue()
    for item in lst:
        QI.put(item)

    def pfn():
        time.sleep(0.001)
        # print(QI.empty(), QI.qsize())
        while QI.qsize() > 0:  # not QI.empty():
            L.acquire()
            item = QI.get()
            L.release()
            obj = fn(item)
            QO.put(obj)

    procs = []
    for nw in range(nworkers):
        P = mp.Process(target=pfn, daemon=True)
        time.sleep(0.001)
        P.start()
        procs.append(P)

    return [QO.get() for i in tqdm(range(len(lst)), "Mapping in parallel", smoothing=0.3)]

def weighted_choice (lst, n, weights):
    probs = weights/np.sum(weights)
    return np.random.choice(lst, n, False, probs)

def softmax (x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def augment_smile (sm):
    mol = Chem.MolFromSmiles(sm)
    return Chem.MolToSmiles(mol, doRandom=True)

def canonize_smile (sm):
    m = Chem.MolFromSmiles(sm)
    try: return Chem.MolToSmiles(m, canonical=True)
    except: return None
