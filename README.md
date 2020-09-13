# LegoGram - molecular grammars for the generation of chemical structures by deep learning

Molecular grammar is a new chemical representation designed to provide faultless generation of organic structures

## Installation 
```
pip install git+https://github.com/sergsb/LegoGram.git
```

## Quick Start 

```
from legogram import LegoGram, pretrained
model = LegoGram(load=pretrained.M_250k)
```

## Encode and decode

```
mol = 'COC(=O)Cc1csc(NC(=O)Cc2coc3cc(C)ccc23)n1'
encoded = model.encode(mol)`
model.decode(encoded) == "COC(=O)Cc1csc(NC(=O)Cc2coc3cc(C)ccc23)n1" #It's a toy example. In production compare by InChI
True

```

## Create you own grammar

```
model = legogram.LegoGram(smiles="legogram/data/250k_rndm_zinc_drugs_clean.smi", optimize_limit=100)
```



*Check source code for documentation: "legogram.py" for base interface, "apps/" for usage examples.* 

