from legogram.base import *

class LegoGramRNNSampler():
    def __init__ (self, lg):
        self.lg = lg
        self.special_tokens = ['<pad>', '<bos>', '<eos>']
        self.nspecs = len(self.special_tokens)
        self.vocsize = self.lg.vocsize+self.nspecs
    
    def encode (self, smi):
        return [c+self.nspecs for c in self.lg.encode(smi)]
    
    def decode (self, code):
        return self.lg.decode([c - self.nspecs for c in code])
        #return self.lg.decode([c-self.nspecs for c in code])
    
    def get_compat_rules (self, graph): # mask only
        rule_mask = self.lg.get_compat_rules(graph, as_mask=True)
        return rule_mask#np.hstack((np.ones((self.nspecs), np.float32), rule_mask))
    
    def init_state (self, batch_size):
        graphs = [None] * batch_size
        finished = [False] * batch_size
        return (graphs, finished)
    
    def sample(self, state, logits, placeholder_idx):
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
                    current[i] = placeholder_idx
                    continue
                
                logits_of_a_rule = logits[i] * mask
                bool_mask = mask.astype(np.bool)
                prob = softmax(logits_of_a_rule[bool_mask])
                current[i] = np.random.choice(np.array(range(logits.shape[-1]))[bool_mask], 1, p=prob)[0]
                if not check_compat(graphs[i], self.lg.rules[current[i]]):
                    raise Exception(
                        "Failed to combine rules! {} and {}".format(graphs[i], self.lg.rules[current[i]]))
                graphs[i] = combine_rules(graphs[i], self.lg.rules[current[i]])
            
            elif (not graph):
                prob = softmax(logits[i])
                current[i] = np.random.choice(range(logits.shape[-1]), 1, p=prob)[0]
                graphs[i] = self.lg.rules[current[i]]
            else:
                current[i] = placeholder_idx
        state = (graphs, finished)
        return state, [c + self.nspecs for c in current]

