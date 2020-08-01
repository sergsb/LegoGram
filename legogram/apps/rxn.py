from legogram.base import *

class LegoGramRXN ():
    def __init__ (self, lg):
        self.lg = lg
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '>', '.']
        self.nspecs = len(self.special_tokens)
        self.vocsize = self.lg.vocsize+self.nspecs
        
    def encode (self, mix):
        def _encode (smi):
            return [c+self.nspecs for c in self.lg.encode(smi)]
        
        res = []
        for i,submix in enumerate(mix.split('>')):
            if i > 0:
                res += [4] # 4 for '>'
            for j,sm in enumerate(submix.split('.')):
                if j > 0:
                    res += [5] # 5 for '.'
                if len(sm) > 0:
                    res += _encode(sm)
        return res
    
    def decode (self, mix):
        def _decode (code):
            return self.lg.decode([c-self.nspecs for c in code])
        
        res = ""
        start = 0
        for i,token in enumerate(mix):
            if token in [5, 4]: # '.', '>'
                res += _decode(mix[start:i])
                res += self.special_tokens[token]
                start = i+1
        if i >= start:
            res += _decode(mix[start:i+1])
        return res
