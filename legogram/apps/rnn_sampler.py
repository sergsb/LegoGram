from legogram.base import *
import numpy as np 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class LegoGramRNNSampler():
    def __init__ (self, lg,optimize=True):
        self.lg = lg
        self.optimize = optimize
        self.special_tokens = ['<pad>', '<bos>', '<eos>']
        self.nspecs = len(self.special_tokens)
        self.vocsize = self.lg.vocsize+self.nspecs

    def encode (self, smi):
        return [c+self.nspecs for c in self.lg.encode(smi,optimize=self.optimize)]
    
    def decode (self, code):
        return self.lg.decode([c - self.nspecs for c in code])
    
    def get_compat_rules (self, graph): # mask only
        rule_mask = self.lg.get_compat_rules(graph, as_mask=True)
        return rule_mask
    
    def init_state (self, batch_size):
        graphs = [None] * batch_size
        finished = [False] * batch_size
        return (graphs, finished)
    
    def sample(self, state, logits, placeholder_idx):
        placeholder_idx=[-3,-2,-1]
        graphs, finished = state
        batch_len = logits.shape[0]
        logits = logits[:,self.nspecs:]
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
                if not check_compat(graphs[i].graph, list(self.lg.rules.keys())[list(self.lg.rules.values()).index(current[i])].graph):
                    raise Exception(
                        "Failed to combine rules! {} and {}".format(graphs[i], list(self.lg.rules.keys())[list(self.lg.rules.values()).index(current[i])]))
                graphs[i] = Rule(combine_rules(graphs[i].graph, list(self.lg.rules.keys())[list(self.lg.rules.values()).index(current[i])].graph))
            
            elif (not graph):
                prob = softmax(logits[i])
                current[i] = np.random.choice(range(logits.shape[-1]), 1, p=prob)[0]
                graphs[i] = list(self.lg.rules.keys())[list(self.lg.rules.values()).index(current[i])]
            else:
                current[i] = placeholder_idx[0]
        state = (graphs, finished)
        return state, np.array([c + self.nspecs for c in current])
