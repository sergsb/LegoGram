from legogram.base import *
from legogram.grammar import nt_fingerprint

def rule_signature (rule):
    nt_list = list(rule.graph.vs.select(name="NT"))
    nt_fps = tuple([tuple(nt_fingerprint(rule.graph, nt)) \
                    for nt in nt_list])
    income = tuple(sorted(rule.graph.vs[0]['income']))
    sgn = tuple((nt_fps, income))
    return sgn

class LegoGramMutator ():
    def __init__ (self, lg):
        self.lg = lg
        self.calc_ichblt()
        self.fails = []
        
        self.charges = {}
        for i,r in enumerate(self.lg.rules):
            names = "".join(r.graph.vs['name'])
            self.charges[r] = names.count("+")-names.count("-")
    
    def calc_ichblt (self):
        ichblt = dict()
        for i,rule in enumerate(self.lg.rules):
            sgn = rule_signature(rule)
            if sgn in ichblt.keys():
                ichblt[sgn].add(i)
            else:
                ichblt[sgn] = {i}
        self.ichblt = ichblt
    
    def mutate (self, smi, n=1):
        try:
            code = self.lg.encode(smi)
            
            replaces = []
            for code_id,rule_id in enumerate(code):
                rule = self.lg.rules_back[rule_id]
                sgn = rule_signature(rule)
                if sgn in self.ichblt:
                    for rule2_id in self.ichblt[sgn]:
                        if rule2_id != rule_id:
                            rule2 = self.lg.rules_back[rule2_id]
                            freq  = self.lg.freqs[rule]
                            freq2 = self.lg.freqs[rule2]
                            score = (freq*freq2)**(1/4)
                            if self.charges[rule] != self.charges[rule2]:
                                score /= 2
                            replaces.append([code_id, rule_id, rule2_id, score])
            
            replaces = sorted(replaces, key=lambda x: -x[3])
            replaces = np.array(replaces)
            repl_ids = weighted_choice(np.arange(replaces.shape[0]), n, replaces[:,3])
            replaces = replaces[repl_ids.tolist()][:,:3]
            replaces = np.round(replaces).astype(np.int)
            
            for code_id, rule_id, rule2_id in replaces:
                code[code_id] = rule2_id
            smi2 = self.lg.decode(code)
            return smi2
        
        except Exception as e:
            self.fails.append([smi, str(e)])
            return smi

