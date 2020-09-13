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
>>True

```


## Encode with compression

```
mol = 'COC(=O)Cc1csc(NC(=O)Cc2coc3cc(C)ccc23)n1'
noncompressed = model.encode(mol)
compressed    = model.encode(mol,optimize=True)
print("SMILES len = {}, uncomressed grammar = {}, comressed grammar = {}".format(len(mol),len(noncompressed),len(compressed))
>> SMILES len = 40, uncomressed grammar = 26, comressed grammar = 14

```

## Create you own grammar

```
model = legogram.LegoGram(smiles="legogram/data/250k_rndm_zinc_drugs_clean.smi", optimize_limit=100)
```



*Check source code for documentation: "legogram.py" for base interface, "apps/" for usage examples.* 

