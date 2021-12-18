# LegoGram - molecular grammars for the generation of chemical structures by deep learning

Molecular grammar is a new chemical representation designed to provide faultless generation of organic structures

## Installation 
```
pip install git+https://github.com/sergsb/LegoGram.git
```

## Quick Start 

```
from legogram import LegoGram, pretrained
model = LegoGram(load=pretrained.M_250k_kekulized_grammar)
```

## Encode and decode

```
encoded = model.encode("COC(=O)CC1=CSC(NC(=O)CC2=COC3=CC(C)=CC=C23)=N1")
model.decode(encoded) == "COC(=O)CC1=CSC(NC(=O)CC2=COC3=CC(C)=CC=C23)=N1" #It's a toy example. In production compare by InChI
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



*Check source code for documentation: "legogram.py" for base interface, "apps/" for usage plugins.* 
### Docs and Citation 
This library was described in my [PhD Thesis](https://www.skoltech.ru/app/data/uploads/2020/12/thesis3.pdf) (Chapter 6). 
Sosnin, Sergey (2021): Exploration of Chemical Space by Machine Learning. figshare. Thesis. https://doi.org/10.6084/m9.figshare.14160683.v1 
