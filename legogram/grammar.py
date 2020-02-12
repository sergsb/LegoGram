from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig


###############################################################
### utils

def canonize_smile(sm):
    m = Chem.MolFromSmiles(sm)
    try:
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except:
        return None


def bond2rank(bond_type):
    return {'S': 0, 'D': 1, 'T': 2, 'A': 3}[str(bond_type)[0]]


def rank2bond(bond_rank):
    try:
        return [Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC][bond_rank]
    except:
        return Chem.BondType.UNSPECIFIED


def rank2bond_char(bond_rank):
    char0 = str(rank2bond(bond_rank))[0]
    return {'S': '-', 'D': '=', 'T': '#', 'A': '()'}[char0]


def edge(g, frm, to):  # maybe switch to g.get_eid(frm,to)
    return list(g.es.select(_between=([frm], [to])))[0]


def mol2graph(mol):
    G = ig.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_vertex(atom.GetSmarts())
    for bond in mol.GetBonds():
        i1 = bond.GetBeginAtomIdx()
        i2 = bond.GetEndAtomIdx()
        G.add_edge(i1, i2, bond=bond2rank(bond.GetBondType()))
    return G


def graph2mol(graph):
    emol = Chem.RWMol()
    for v in graph.vs():
        emol.AddAtom(Chem.AtomFromSmiles(v["name"]))
    for e in graph.es():
        emol.AddBond(e.source, e.target, rank2bond(e['bond']))
    mol = emol.GetMol()
    return mol


def draw(g, out=None):
    def vcolor(i):
        v = g.vs[i]
        if i == 0:
            return 'rgb(200,127,127)'
        elif v['name'] == "NT":
            return 'rgb(200,200,200)'
        else:
            return 'rgb(255,255,255)'

    def vname(i):
        v = g.vs[i]
        if i == 0:
            if 'income' in v.attribute_names():
                income_chars = [rank2bond_char(inc) for inc in g.vs[0]['income']]
                return v['name'] + "\n" + ",".join(income_chars)
            else:
                return v['name']  # +"\n"+"({})".format(i)
        elif v['name'] == "NT":
            return "NT\n" + str(v['order'])  # +"\n"+"({})".format(i)
        else:
            return v['name']  # +"\n"+"({})".format(i)

    def vname2(i):
        v = g.vs[i]
        if 'income' in v.attribute_names():
            income_chars = [rank2bond_char(inc) for inc in v['income']]
        else:
            income_chars = ""
        if i == 0:
            return v['name'] + "\n" + "({})".format(i) + ",".join(income_chars)
        elif v['name'] == "NT":
            return "NT\n" + str(v['order']) + "\n" + "({})".format(i) + ",".join(income_chars)
        else:
            return v['name'] + "\n" + "({})".format(i) + ",".join(income_chars)

    def ewidth(i):
        e = g.es[i]
        if 'bond' in e.attribute_names() and e['bond'] > 0:
            return 6
        else:
            return 2

    def elabel(i):
        e = g.es[i]
        if 'bond' in e.attribute_names() and e['bond'] > 0:
            return rank2bond_char(e['bond'])
        else:
            return ""

    dim = len(g.vs) * 15 + 225
    ig.plot(g, "/tmp/out.png" if out is None else out,
            layout="fruchterman_reingold",
            bbox=(0, 0, dim, dim),
            vertex_label=[vname2(i) for i in range(len(g.vs))],
            vertex_size=50,
            vertex_label_size=20,
            vertex_color=[vcolor(i) for i in range(len(g.vs))],
            edge_width=[ewidth(i) for i in range(len(g.es))],
            edge_label=[elabel(i) for i in range(len(g.es))],
            edge_label_size=20,
            # edge_width=10,
            margin=60)
    # if out is None:
    #    !feh /tmp/out.png


################################################################

def ring_info(g):
    rings = [tuple(sorted(sub)) for sub in g.biconnected_components() if len(sub) > 2]
    ring_dict = {}
    for i, r in enumerate(rings):
        for v in r:
            if ring_dict.get(v) is None:
                ring_dict[v] = [i]
            else:
                ring_dict[v].append(i)
    return rings, ring_dict


def _make_ring_rule(g, atom, parents, ring):
    rule = g.subgraph(ring)
    for i, v in enumerate(rule.vs):
        v['name'] = "NT"
        v['order'] = i
        v['income'] = []

    income = [edge(g, p, atom)['bond'] for p in parents]
    rule.vs[0]['income'] = income
    return rule, [v.index for v in g.vs.select(ring)]


def make_ring_rule(g, start_atom, start_parents, ring):
    rule, ring_atoms = _make_ring_rule(g, start_atom, start_parents, ring)
    edges_to_go = []

    for atom in ring_atoms:
        neibs = [v.index for v in g.vs[atom].neighbors()]
        ring_parents = list(set(neibs) & set(ring_atoms))
        if atom == start_atom:
            ring_parents = list(set(ring_parents) | set(start_parents))
        edges_to_go += [[ring_parents, atom]]
    return rule, edges_to_go


def make_rule(g, atom, parents):
    # parents - list of vertex ids in original graph 'g'
    rule = ig.Graph()
    income = [edge(g, p, atom)['bond'] for p in parents]  # list of bond type incoming to current atom
    rule.add_vertex(g.vs[atom]['name'], income=income)

    neibs = [a.index for a in g.vs[atom].neighbors()]  # atom neibs in orig graph
    children = [a for a in neibs if a not in parents]

    next_moves = []
    for i, ch in enumerate(children):
        rule.add_vertex("NT", order=i, income=[])
        rule.add_edge(0, i + 1, bond=edge(g, atom, ch)['bond'])
        next_moves.append([[atom], ch])
    return rule, next_moves


def encode(g):
    if type(g) is str:
        mol = Chem.MolFromSmiles(g)
        g = mol2graph(mol)

    ri = ring_info(g)
    ri_list, ri_dict = ri
    visited_rings = []
    rules = []
    edges_to_go = [[[], 0]]

    infloop_protect = len(g.vs) * 2 + 100
    while len(edges_to_go) > 0:
        if infloop_protect <= 0:
            raise Exception("Encode Inf loop " + str(Chem.MolToSmiles(graph2mol(g))))
        infloop_protect -= 1

        parents, atom = edges_to_go.pop()
        ring_ids = ri_dict.get(atom)

        ok = False
        if ring_ids is not None:  # atom in at least one ring
            if len(ring_ids) == 1:
                ring_id = ring_ids[0]
                if ring_id not in visited_rings:
                    rule, next_moves = make_ring_rule(g, atom, parents, ri_list[ring_id])
                    visited_rings.append(ring_id)
                    ok = True
            elif len(ring_ids) == 2:
                id1, id2 = ring_ids
                ring_id = None
                if id1 not in visited_rings and id2 in visited_rings:  # visit id1
                    ring_id = id1
                elif id1 not in visited_rings and id2 in visited_rings:  # visit id2
                    ring_id = id2
                elif id1 not in visited_rings and id2 not in visited_rings:
                    # both rings not visited, probably will never happen
                    # choose any ring with highest atom id (excluding current atom)
                    # why highest???? idk it just works
                    min1 = min(set(ri_list[id1]) - set([atom]))
                    min2 = min(set(ri_list[id2]) - set([atom]))
                    if min1 > min2:
                        ring_id = id1
                    else:
                        ring_id = id2

                if ring_id is not None:
                    rule, next_moves = make_ring_rule(g, atom, parents, ri_list[ring_id])
                    visited_rings.append(ring_id)
                    ok = True

            else:  # len(ring_ids) > 2
                # wtf is this mol?
                raise Exception("wtf is this mol #1 " + str(Chem.MolToSmiles(graph2mol(g))))

        if not ok:
            rule, next_moves = make_rule(g, atom, parents)

        rules.append(rule)
        edges_to_go += next_moves[::-1]
    return rules


def nt_fingerprint(g, nt_id):  # list of edge types for given nt (plus own income if any)
    own_income = g.vs[nt_id.index]['income']
    income = g.es[g.incident(nt_id)]['bond']
    return sorted(own_income + income)


def check_compat(r1, r2):
    nt_list = list(r1.vs.select(name="NT"))
    nt_fps = [nt_fingerprint(r1, nt) for nt in nt_list]
    income = sorted(r2.vs[0]['income'])
    return (income in nt_fps)


def combine_rules(r1, r2):
    r1 = r1.copy()
    # income: list of bond types
    # find NT with exactly same bond amount and types, and with lowest order
    income = sorted(r2.vs[0]['income'])
    nt_list = list(r1.vs.select(name="NT"))  # .select(_degree = len(income)))
    nt_list = [v for v in nt_list if v.degree() + len(v['income']) == len(income)]
    # def nt_fingerprint (g, nt_id): # list of edge types for given nt (plus own income if any)
    #    own_income = g.vs[nt_id.index]['income']
    #    income = g.es[g.incident(nt_id)]['bond']
    #    return sorted(own_income+income)

    nt_list = list(filter(lambda nt_id: nt_fingerprint(r1, nt_id) == income, nt_list))
    nt_id = sorted(nt_list, key=lambda x: x['order'])[0].index
    # nt_id in r1 is vertex0 in r2. unite!
    orig_order = r1.vs[nt_id]['order']  # dont overwrite nt order if entering a ring
    r1.vs[nt_id]['name'] = r2.vs[0]['name']
    if r1.vs[nt_id]['name'] != "NT":
        r1.vs[nt_id]['order'] = None
        r1.vs[nt_id]['order'] = orig_order

    # todo: make vertex map as dict without numpy
    vertex_inc = len(r1.vs) - 1
    vertex_map = np.tile(r2.vs.indices, (2, 1)).T
    vertex_map[:, 1] += vertex_inc
    vertex_map[0][1] = nt_id
    vertex_map = dict(vertex_map)

    # inc nt orders in first rule by number of nt's in second rule
    # -1 for current NT which is closed after connection
    nt_order_inc = len(r2.vs.select(name="NT")) - 1
    for v in r1.vs.select(name="NT"):
        v['order'] += nt_order_inc

    for v in r2.vs:
        if v.index != 0:
            r1.add_vertex(**v.attributes())
        else:
            if v['name'] == "NT":  # this is ring enter, should be 0 (previously was increased by nt_order_inc)
                r1.vs[nt_id]['order'] = 0

    for e in r2.es:
        v1, v2 = e.tuple
        r1.add_edge(vertex_map[v1], vertex_map[v2], **e.attributes())
    return r1


def decode(rules):
    res = rules[0]
    for rule in rules[1:]:
        res = combine_rules(res, rule)
    return Chem.MolToSmiles(graph2mol(res))


def _test(sm):
    sm = canonize_smile(sm)
    return sm == decode(encode(sm))


from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
DrawingOptions.atomLabelFontSize = 15
# DrawingOptions.dotsPerAngstrom = 150
# DrawingOptions.bondLineWidth = 1
DrawingOptions.useFraction = 0.99


def rdraw(mol, path="/tmp/out.png"):
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    Draw.MolToFile(mol, path, size=(500, 300), fitImage=True, includeAtomNumbers=False)


### test
'''
sm = "CC(C)(C)OC(=O)NC1(c2ccc(-c3c(-c4nc5ccccc5s4)nc4n3-c3cccnc3Nc3ccccc3-4)cc2)CCC1"
res = encode(sm)
'''

################################################################

def rule2descr(rule):
    rule = rule.copy()
    for v in rule.vs:
        if v['name'] == "NT":
            v['name'] += str(v['order'])
            v['order'] = None
        if v.index == 0:
            v['name'] = "0" + v['name']
        v['name'] += ":" + ''.join([str(x) for x in v['income']])
        v['income'] = []
    return rule


def rule_eq(r1, r2):
    r1d, r2d = rule2descr(r1), rule2descr(r2)
    permutations = r1d.get_isomorphisms_vf2(r2d,
                                            node_compat_fn=node_eq,
                                            edge_compat_fn=edge_eq)
    return len(permutations) > 0


def node_eq(g1, g2, ni1, ni2):
    return g1.vs[ni1]['name'] == g2.vs[ni2]['name']


def edge_eq(g1, g2, ei1, ei2):
    return g1.es[ei1]['bond'] == g2.es[ei2]['bond']


def ring_connectivity(mol):
    ssr = Chem.GetSymmSSSR(mol)
    rings = [set(x) for x in ssr]
    n = len(rings)
    adj = np.zeros((n, n), np.int)
    for i in range(n):
        for j in range(n):
            if i != j:
                common_atoms = rings[i] & rings[j]
                if len(common_atoms) >= 2:
                    adj[i, j] = 1
    return adj


def ring_compliexity_filter(mol, thr=2):
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    adj = ring_connectivity(mol)
    # how many other ring have at least one common edge with each one
    nconns = np.sum(adj, axis=0)
    return np.sum(nconns >= thr) == 0
