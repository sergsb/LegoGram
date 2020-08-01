
from .grammar import combine_rules, check_compat, encode, decode, graph2mol, nt_fingerprint, rule_eq, draw
from .base import * #LegoGram
from .utils import *
from .apps import LegoGramMutator, LegoGramRXN, LegoGramRNNSampler

from pkg_resources import resource_filename
import glob
import os
from enum import Enum

_model_dir = resource_filename(__name__, "models")
_model_paths = glob.glob(os.path.join(_model_dir, "*.pkl"))

def assign_name (path):
    name = os.path.splitext(os.path.split(path)[1])[0]
    if len(name) == 0 or name[0].isdigit():
        name = "M_"+name
    return name
    
_model_repo = dict([(assign_name(path), path) for path in _model_paths])

def enum(**enums): return type('Enum', (), enums)

pretrained = enum(**_model_repo)

    
