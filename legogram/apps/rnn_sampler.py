from legogram.base import *
import numpy as np 
from enum import IntEnum 

class SpecialTokens(IntEnum):
    PAD = 0 #Padding
    BOS = 1 #Begin of Sequence
    EOS = 2 #End of Sequence
      
class LegoGramRNNSampler():
    def __init__(self, lg, optimize=True):
        self.lg = lg
        self.optimize = optimize
        self.nspecs = len(SpecialTokens)
        self.vocsize = self.lg.vocsize + self.nspecs

    def encode(self, smi):
        return [c + self.nspecs for c in self.lg.encode(smi, optimize=self.optimize)]

    def decode(self, code):
        res = []
        for c in code:
            if c >= self.nspecs:
                res.append(c - self.nspecs)
        print("res is ", res)
        return self.lg.decode(res)

    def get_compat_rules(self, graph):  # mask only
        rule_mask = self.lg.get_compat_rules(graph, as_mask=True)
        return rule_mask

    def init_state(self, batch_size):
        graphs = [None] * batch_size
        finished = [False] * batch_size
        return (graphs, finished)

    def sample(self, state, logits):
        placeholder_idx = [-3, -2, -1]
        graphs, finished = state
        batch_len = logits.shape[0]
        logits = logits[:, self.nspecs:]
        current = [None] * batch_len
        for i in range(len(graphs)):
            graph = graphs[i]
            if graph and (not finished[i]):
                mask = self.get_compat_rules(graph)
                if mask.sum() == 0.:
                    finished[i] = True
                    current[i] = placeholder_idx[2]
                    continue
                logits_of_a_rule = logits[i] * mask
                bool_mask = mask.astype(np.bool)
                prob = softmax(logits_of_a_rule[bool_mask])
                current[i] = np.random.choice(np.array(range(logits.shape[-1]))[bool_mask], 1, p=prob)[0]
                if not check_compat(graphs[i].graph,
                                    list(self.lg.rules.keys())[list(self.lg.rules.values()).index(current[i])].graph):
                    raise Exception(
                        "Failed to combine rules! {} and {}".format(graphs[i], list(self.lg.rules.keys())[
                            list(self.lg.rules.values()).index(current[i])]))
                graphs[i] = Rule(combine_rules(graphs[i].graph, list(self.lg.rules.keys())[
                    list(self.lg.rules.values()).index(current[i])].graph))

            elif (not graph):
                prob = softmax(logits[i])
                current[i] = np.random.choice(range(logits.shape[-1]), 1, p=prob)[0]
                graphs[i] = list(self.lg.rules.keys())[list(self.lg.rules.values()).index(current[i])]
            else:
                current[i] = placeholder_idx[0]
        state = (graphs, finished)
        return state, np.array([c + self.nspecs for c in current])
