from os import cpu_count
import logging
logger = logging.getLogger(__name__)
import os
if os.environ.get('LOGGING_LEVEL'):
    logging.basicConfig(level=int(os.environ.get('LOGGING_LEVEL')))
else:
    logging.basicConfig(level=logging.CRITICAL)


try:
    from mpi4py import MPI
    MPI_AVAIL = True
except:
    MPI_AVAIL = False
    print("Will not use MPI, because `mpi4py` package not found.")

from .grammar import combine_rules, check_compat, encode, decode, graph2mol, nt_fingerprint, rule_eq, draw
from .utils import *

class Rule():
     def __init__(self, rule):
         "Rule is a wrapper class for grammar graphs. Implements compare and drawing methods"
         self.graph = rule
         self._recalculate_rule_hash()

     def __eq__(self, other):
         "Compare underlying graphs"
         return rule_eq(self.graph, other.graph)

     def _recalculate_rule_hash(self):
        self.cahced_hash = hash((self.graph.vcount(),
                                 self.graph.ecount(),
                                 tuple(self.graph.vs[0]['income']),
                                 tuple(self.graph.vs['name']),
                                 tuple(self.graph.es['bond']) if len(self.graph.es) > 0 else None))

     def __hash__(self):
         if hasattr(self,'cahced_hash'):
            return self.cahced_hash
         else:
             self._recalculate_rule_hash()
             return self.cahced_hash

     def draw (self, path=None, with_order=False, neato_seed=None,format=None):
         """
         Draw rule graph.
         If `path` is not None, use it as filepath to save png image.
         If `with_order` is True, display order number of each node.
         Set `neato_seed` to some integer - random seed for neato optimizer, if you do not like the result (overlaps are possible)
         Returns pygraphviz object.
         """
         if not format:
            return draw(self.graph, path, with_order, neato_seed)
         else:
            return draw(self.graph, path, with_order, neato_seed).draw(format=format,prog='neato')

     def _repr_png_(self):
         return draw(self.graph, None, with_order=False, neato_seed=None).draw(format='png', prog='neato')

class LegoGram ():
    def __init__ (self, load=None, smiles=None, optimize=True, mpi=False, canonical=True, optimize_limit=1000, nworkers=-1, maxpart=1000):
        """
        LegoGram is the interface to the molecular grammar. Main methods to use are `encode` and `decode` (see below).
        Init keys:
        
        `load`: if not None, use as path to a state dict of previosly built LegoGram object.
        Some prebuilt models could be found at `legogram/models` directory. The list of currently avaliable models is gathered in `pretrained` enum object.
        That means you can do
        > LegoGram(load=pretrained.M_250k) # (M_ is auto added when filename starts with a digit)

        `smiles`: Is used to build a new grammar (or update a loaded one - see below).
        It could be a list of smiles strings, or a filename string (smiles are taken from the file in that case)
        Some datasets could be found at 'legogram/data' directory.
        `smiles` option could be given along with `load`. In that case a pretrained model is loaded, and after that is extended with new data, given as `smiles`

        `optimize`: If True, perform grammar optimization. The main idea is to find most common rule patterns, and combine them into new rules.
        Only available, if `smiles` is given.

        `optimize_limit`: How many (most frequent) rule patterns to select in optimization step.

        `canonical`: If True (default), convert each smiles to the canonical form.

        `mpi`: Set True, if building a new grammar seems too long in your case. Also do not forget to install `mpi4py` package, and launch python with `mpiexec -n` prefix.

        """
        self.canonical = canonical
        self.optimize_limit = optimize_limit
        self.maxpart = 1000
        self.nworkers = nworkers if nworkers > 0 else cpu_count()
        logger.info("Using {} workers".format(self.nworkers))
        if load is not None:
            self.__dict__.update(joblib.load(load))
        else:
            self.freqs = Counter()
            self.fails = []
            
        
        if smiles is not None:
            self.freqs = Counter()
            self.fails = []
            if type(smiles) is str:
                with open(smiles) as fh:
                    smiles = fh.read().split('\n')[:-1]
            if mpi and MPI_AVAIL:
                self.more_smiles_mpi(smiles)
            else:
                self.more_smiles(smiles)

            if optimize:
                print(f"Optimizing grammar to have {optimize_limit} more rules")
                self.optimize(smiles, self.optimize_limit)

    def encode (self, sm, optimize=False):
        """
        Converts given smiles to a series of numbers (indices of corresponding rules).
        Is self.canonical is True, the given smiles is canonicalized.
        
        If `optimize` is True, an optimized encoding is returned (trying to substitute registered patterns with their codes)
        """        
        rules = [Rule(r) for r in encode(self.prepare_smile(sm))]
        try:
            code = [self.rules[r] for r in rules]
        except:
            raise Exception("Rule not found for smiles: " + sm)
        
        if not optimize:
            return code
        
        else:
            if not self.is_optimized():
                raise Exception("This instance of grammar has not been optimized")
            return self.optimize_encoding(code)

    def decode (self, code, partial=False):
        """
        Converts given encoding (series of numbers) to smiles.
        if `partial` is True, returns a rule instead of smiles,
        meaning that the code may not represent a complete molecule (partial code).
        """
        rules = []
        for i in code:
            rules.append(self.rules_back[i].graph)
        result = decode(rules, partial)
        if partial:
            result = Rule(result)
        return result
                
    def save (self, path):
        "Save model state dict with given path"
        joblib.dump(self.__dict__, path)

    def is_optimized (self):
        "If optimized rules are calculated for this grammar"
        return hasattr(self, 'replace_table')

        
    def get_compat_rules (self, rule, as_mask=False):
        """
        Returns subset of rules, compartible with the given rule.
        Any rule can be checked, even not existing in the grammar dictionary
        """
        nt_list = list(rule.graph.vs.select(name="NT"))
        nt_fps = [tuple(nt_fingerprint(rule.graph, nt)) for nt in nt_list]
        rule_ids = []
        for isect in self.compat.keys() & nt_fps:
            rule_ids += self.compat[isect]
        if as_mask:
            mask = np.zeros(self.vocsize, np.float32)
            mask[rule_ids] = 1.
            return mask
        else:
            return np.array(rule_ids)

    def most_common (self, n_base=None, n_opt=None):
        """
        Returns a reduced copy of grammar, with only `n_base` basic rules, and `n_opt` optimized rules.
        If `n_base` is None, basic rules are not reduced, same for `n_opt`
        """
        if n_base is None and n_opt is None:
            raise Exception("Provide at least one parameter to reduce")
        if n_base is None:
            n_base = len(self.freqs)
            if self.is_optimized():
                n_base -= len(self.replace_table)
        if n_opt is None:
            n_opt = len(self.replace_table)
        
        grammar = copy.deepcopy(self)
        
        base_freqs = OrderedDict()
        opt_freqs = OrderedDict()
        base_count = 0
        opt_count = 0
        if grammar.is_optimized():
            replace_back = {id:seq for (seq,id) in grammar.replace_table.items()}
            replace_table = dict()

        for rule,code in grammar.rules.items():
            freq = grammar.freqs[rule]
            if grammar.is_optimized() and \
               code in grammar.replace_table.values():
                if opt_count < n_opt:
                    opt_freqs[rule] = freq
                    opt_count += 1
                    replace_table[replace_back[code]] = code
            else:
                if base_count < n_base:
                    base_freqs[rule] = freq
                    base_count += 1

        if grammar.is_optimized():
            base_freqs.update(opt_freqs)
            grammar.replace_table = replace_table

        grammar.freqs = base_freqs
        grammar.finalize_smiles()
        return grammar

        
    ###############################################################
    ### internal stuff
        
    def unique_rules (self, smiles):
        freqs = Counter()
        for sm in smiles:
            rules = [Rule(r) for r in encode(self.prepare_smile(sm))]
            freqs.update(rules)
        return freqs
    
    def more_smiles (self, smiles):
        parts = []
        for i in range(len(smiles)//self.maxpart+1):
            parts.append(smiles[i*self.maxpart:(i+1)*self.maxpart])

        print("Encoding smiles")
        parts = map_parallel(parts, self.unique_rules, self.nworkers)
        
        for p in tqdm(parts, "Gathering rules"):
            self.freqs.update(p)

        self.freqs = OrderedDict(self.freqs.most_common())
        self.finalize_smiles()
                    
    def more_smiles_mpi (self, smiles):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank == 0:
            t = tqdm(total=len(smiles), desc="Generating rules")

        scattered = comm.scatter(split(smiles,size), root=0)
        for i,sm in enumerate(scattered):
            self.freqs.update(self.unique_rules([sm]))
            if_print = i % 100 == 0 and i>0 or i == len(scattered)-1
            if if_print:
                ready = comm.gather(i,root=0)
                if rank == 0:
                    t.update(sum(ready) - t.n)
                    t.refresh()
        gathered = comm.gather(self.freqs, root=0)
        if rank == 0:
            t.close()
            self.freqs = Counter()
            for result in tqdm(gathered,desc="Gathering rules"):
                self.freqs.update(result)
                
        self.freqs = OrderedDict(self.freqs.most_common())
        self.finalize_smiles()
        
    def calc_compat (self):
        compat = {}
        for rule,code in self.rules.items():
            rule_hash = tuple(rule.graph.vs[0]['income'])
            if rule_hash not in compat.keys():
                compat[rule_hash] = {code}
            else:
                compat[rule_hash].add(code)
        self.compat = compat
        
    def finalize_smiles (self):
        self.rules = {r:i for i,r in enumerate(self.freqs.keys())}
        self.rules_back = {i:r for i,r in enumerate(self.freqs.keys())}
        self.vocsize = len(self.rules)
        self.calc_compat()
   

    def prepare_smile (self, sm):
        m = Chem.MolFromSmiles(sm)
        # grammar encode can also deal with rdkit Mol objects
        # so avoid extra calculations
        return Chem.MolToSmiles(m, canonical=self.canonical, isomericSmiles=False)
    
    def rules_combinations_ (self, smiles):
        combinations = Counter()
        for sm in smiles:
            code = self.encode(sm)
            subseqs = []
            for i in range(len(code)-1):
                for j in range(i+2, len(code)):
                    subseqs.append( tuple(code[i:j]) )
            combinations.update(Counter(subseqs))
        return OrderedDict(combinations.most_common(self.optimize_limit*10))

    def rules_combinations (self, smiles):
        parts = []
        for i in range(len(smiles)//self.maxpart+1):
            parts.append(smiles[i*self.maxpart:(i+1)*self.maxpart])

        parts = map_parallel(parts, self.rules_combinations_, self.nworkers)
        combinations = Counter()
        for p in parts:
            combinations.update(p)
        return OrderedDict(combinations.most_common())
    
    def optimize (self, smiles, limit):
        combs = self.rules_combinations(smiles)
        freqs = dict()
        replaces = dict()
        t = tqdm(total=limit, desc="Registering valid combinations")
        for seq,freq in combs.items():
            try:
                rule = self.decode(seq, partial=True)
                freqs[rule] = freq
                replaces[rule] = seq
                t.update(len(freqs)-t.n)
                t.refresh()
            except:
                pass
            if len(freqs) >= limit:
                t.close()
                break
            
        self.freqs.update(freqs)
        print ("Finalize...")
        self.finalize_smiles()
        
        self.replace_table = dict()
        for rule, seq in replaces.items():
            self.replace_table[seq] = self.rules[rule]

    def optimize_encoding (self, code):
        code = copy.deepcopy(code)
        opts = []
        for i in range(len(code)-1):
            for j in range(i+2, len(code)):
                seq = tuple(code[i:j])
                if seq in self.replace_table.keys():
                    opts.append([i,j])
        
        if len(opts) == 0:
            return code
        
        def has_overlaps (opt):
            cur = opt[0]
            for o in opt[1:]:
                if o[0] < cur[1]:
                    return True
                else:
                    cur = o
            return False
        
        def sort_by_len (opts):
            sorted_ids = np.argsort([to-frm for (frm,to) in opts])[::-1]
            sorted_opts = [opts[i] for i in sorted_ids]
            return sorted_opts
        
        def sort_by_pos (opts):
            sorted_ids = np.argsort([frm for (frm,to) in opts])
            sorted_opts = [opts[i] for i in sorted_ids]
            return sorted_opts
        
        def guess_best_opt (opts):
            opts = sort_by_len(opts)
            chosen_opts = [opts[0]]
    
            for next_opt in opts[1:]:
                maybe_added = sort_by_pos(chosen_opts + [next_opt])
                if not has_overlaps(maybe_added):
                    chosen_opts = maybe_added
            return chosen_opts
                
        best_opt = guess_best_opt(opts)
        for frm,to in best_opt:
            replace_id = self.replace_table[tuple(code[frm:to])]
            code[frm:to] = [replace_id] + [None]*(to-frm-1)
        
        code = [c for c in code if c is not None]
        return code

    
