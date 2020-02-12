import numpy as np
import pickle
from tqdm import tqdm
from legogram.grammar import combine_rules, check_compat, encode, decode, graph2mol, nt_fingerprint, rule_eq
import time
import multiprocessing as mp
from pkg_resources import  resource_stream
from enum import Enum

class LegoGramModel(Enum):
    EMPTY = 0
    ZINC250LIMITED = 1
    ChEMBL = 2
    USPTO = 3

class LegoGram():
    def __init__(self, file, maxpart=10000, nworkers=4):
        self.freqs = {}
        self.maxpart = maxpart
        self.nworkers = nworkers
        with open(file) as f:
            smiles = f.read().split("\n")[:-1]
        self.more_smiles(smiles)
        self.calculate_compatibility_hashes()
        '''
        elif model == LegoGramModel.ZINC250LIMITED:
            self = pickle.load(resource_stream(__name__,"models/250k_unoptimized.bin"))
        '''
        '''
        def (file):
            def strings(lst):
                return [str(x) for x in lst]
            data = pd.read_csv(file, sep=",")
            X = strings(data.get("input"))
            Y = strings(data.get('target'))
            smiles = []
            for mixture in X + Y:
                for sm in mixture.replace('>', '.').split('.'):
                    if sm != '':
                        smiles.append(sm)
            smiles = list(set(smiles))
            return smiles
        '''

    @classmethod
    def load(cls,model):
        if model == LegoGramModel.ZINC250LIMITED:
            return pickle.load(resource_stream(__name__,"models/250k_unoptimized.bin"))
        elif model == LegoGramModel.ChEMBL:
            return pickle.load(resource_stream(__name__, "models/ChEMBL_unoptimized.bin"))

    def init_state(self,batch_size):
        graphs = [None] * batch_size  # Prepare an array for graphs
        finished = [False] * batch_size
        return (graphs, finished)
    def _softmax(self,x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()  # only difference

    def sample(self,state, logits, placeholder_idx):

        graphs, finished = state
        batch_len = logits.shape[0]
        current = [None] * batch_len
        for i in range(len(graphs)):
            graph = graphs[i]
            if graph and (not finished[i]):
                mask = self.get_compatible_rules_mask(graph)
                if mask.sum() == 0.:
                    finished[i] = True
                    current[i] = placeholder_idx
                    continue

                logits_of_a_rule = logits[i] * mask
                bool_mask = mask.astype(np.bool)
                prob = self._softmax(logits_of_a_rule[bool_mask])
                current[i] = np.random.choice(np.array(range(logits.shape[-1]))[bool_mask], 1, p=prob)[0]
                if not check_compat(graphs[i], self.rules[current[i]]):
                    raise Exception(
                        "Failed to combine rules! {} and {}".format(graphs[i], self.rules[current[i]]))
                graphs[i] = combine_rules(graphs[i], self.rules[current[i]])

            elif (not graph):
                prob = self._softmax(logits[i])
                current[i] = np.random.choice(range(logits.shape[-1]), 1, p=prob)[0]
                graphs[i] = self.rules[current[i]]
            else:
                current[i] = placeholder_idx
        state = (graphs, finished)
        return state, current

    def dump(self,filename):
        pickle.dump(self,open(filename,'wb'))

    def calculate_compatibility_hashes(self):
        compatibility = {}
        for i,rule in enumerate(self.rules):
            hash_of_the_rule = tuple(rule.vs[0]['income']) #FixMe: maybe keep tuple here? in rule.vs[0]['income']?
            if hash_of_the_rule not in compatibility.keys():
                compatibility[hash_of_the_rule] = {i}
            else:
                compatibility[hash_of_the_rule].add(i)
        self.compatibility_hashes = compatibility
        #return self.compatibility_hashes

    def get_compatible_rules_mask(self,rule):
        if self.compatibility_hashes:
            nt_list = list(rule.vs.select(name="NT"))
            nt_fps = {tuple(nt_fingerprint(rule, nt)) for nt in nt_list}
            intersections = nt_fps.intersection(self.compatibility_hashes)
            compatible_rules = []
            for intersection in intersections:
                compatible_rules += self.compatibility_hashes[intersection]
            mask = np.zeros(len(self.rules), dtype=np.float32)
            mask[compatible_rules] = 1.
            return mask

    def size(self):
        return len(self.rules)

    def _more_smiles(self, smiles):
        freqs = {}
        inc_nrules = 0
        for i, sm in enumerate(smiles):
            rules = encode(sm)
            for nr in rules:
                items = sorted(freqs.items(), key=lambda x: -x[1])
                add = True
                for rule, freq in items:
                    if rule_eq(nr, rule):
                        freqs[rule] += 1
                        add = False
                        break
                if add:
                    freqs[nr] = 1
                    inc_nrules += 1
        sorted_list = sorted(freqs.items(), key=lambda x: -x[1])
        freqs = dict(sorted_list)
        return freqs

    def more_smiles(self, smiles):
        parts = []
        for i in range(len(smiles) // self.maxpart + 1):
            parts.append(smiles[i * self.maxpart:(i + 1) * self.maxpart])

        parts = map_parallel(parts, self._more_smiles, self.nworkers)
        # parts = [self._more_smiles(smiles)] # single-pass

        for p in tqdm(parts):
            compare_items = sorted(self.freqs.items(), key=lambda x: -x[1])
            for nr, ncount in p.items():

                add = True
                for rule, count in compare_items:
                    if rule_eq(nr, rule):
                        self.freqs[rule] += 1
                        add = False
                        break
                if add:
                    self.freqs[nr] = 1
        sorted_list = sorted(self.freqs.items(), key=lambda x: -x[1])
        self.freqs = dict(sorted_list)
        self.rules = [x[0] for x in sorted_list]
        # self.build_compat()

    def build_compat(self):
        self.compat = np.zeros((len(self.rules), len(self.rules)), np.bool)
        for i, r1 in enumerate(self.rules):
            for j, r2 in enumerate(self.rules):
                self.compat[i, j] = check_compat(r1, r2)

    def encode(self, sm):
        _rules = encode(sm)
        code = []
        for _rule in _rules:
            ok = False
            for i, rule in enumerate(self.rules):
                if rule_eq(rule, _rule):
                    ok = True
                    code.append(i)
                    break
            if not ok:
                raise Exception("Rule not found for smiles: " + sm)
        return code

    def decode(self, code):
        rules = []
        for i in code:
            rules.append(self.rules[i])
        return decode(rules)

    def encode_mixture(self, mix):
        res = []
        for i, submix in enumerate(mix.split('>')):
            if i > 0:
                res += ['>']
            for j, sm in enumerate(submix.split('.')):
                if j > 0:
                    res += ['.']
                res += [str(n) for n in self.encode_mol(sm)]
        return res

    def decode_mixture(self, mix):
        res = ""
        start = 0
        for i, token in enumerate(mix):
            if token in ['.', '>']:
                res += self.decode_mol([int(tok) for tok in mix[start:i]])
                res += token
                start = i + 1
        if i > start:
            res += self.decode_mol([int(tok) for tok in mix[start:i + 1]])
        return res


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

    return [QO.get() for i in tqdm(range(len(lst)))]

