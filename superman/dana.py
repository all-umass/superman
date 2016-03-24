from __future__ import print_function
import os.path
import numpy as np
from six.moves import zip


def convert_to_dana(mineral_names, mineral_ids=None,
                    unk_dana=('0','0','0','0')):
  name_to_number = load_dana_map()
  dana = []
  if mineral_ids is None:
    dana_dtype = list(zip(['klass','type','group','species'], ['|S4']*4))
    for name in mineral_names:
      dana.append(name_to_number.get(name, unk_dana))
  else:
    dana_dtype = list(zip(['klass','type','group','species','ID'], ['|S4']*5))
    for name, id_num in zip(mineral_names, mineral_ids):
      dana.append(name_to_number.get(name, unk_dana) + (str(id_num),))
  return np.array(dana, dtype=dana_dtype).view(np.recarray)


def load_dana_map():
  '''Returns a dict from name -> (class,type,group,species)'''
  return dict(_gen_dana_tuples())


def load_dana_tree():
  '''Returns a nested dict from class -> type -> group -> species -> name'''
  dana_tree = {}  # nested dict of class -> type -> group -> species -> name
  for name, parts in _gen_dana_tuples():
    _add_nested(dana_tree, parts, name)
  return dana_tree


def _gen_dana_tuples():
  file_path = os.path.join(os.path.dirname(__file__), 'dana_numbers.txt')
  with open(file_path) as fh:
    for line in fh:
      try:
        num,name = line.strip().split(' ', 1)
      except ValueError as e:
        print(line)
        raise e
      yield name, tuple(num.split('.'))


def _add_nested(tree, keys, value):
  k = keys[0]
  keys = keys[1:]
  if not keys:
    tree[k] = value
  else:
    if k not in tree:
      tree[k] = {}
    _add_nested(tree[k], keys, value)


dana_class_names = {
    1: 'Native Elements and Alloys',
    2: 'Sulfides, including Selenides and Tellurides',
    3: 'Sulfosalts',
    4: 'Simple Oxides',
    5: 'Oxides Containing Uranium, and Thorium',
    6: 'Hydroxides and Oxides Containing Hydroxyl',
    7: 'Multiple Oxides',
    8: 'Multiple Oxides Containing Niobium, Tantalum, and Titanium',
    9: 'Anhydrous and Hydrated Halides',
    10: 'Oxyhalides and Hydroxyhalides',
    11: 'Halide Complexes; Alumino-flourides',
    12: 'Compound Halides',
    13: 'Acid Carbonates',
    14: 'Anhydrous Carbonates',
    15: 'Hydrated Carbonates',
    16: 'Carbonates Containing Hydroxyl or Halogen',
    17: 'Compound Carbonates',
    18: 'Nitrates',
    19: 'Nitrates Containing Hydroxyl or Halogen',
    20: 'Compound Nitrates',
    21: 'Anhydrous and Hydrated Iodates',
    22: 'Iodates Containing Hydroxyl or Halogen',
    23: 'Compound Iodates',
    24: 'Anhydrous Borates',
    25: 'Anhydrous Borates Containing Hydroxyl or Halogen',
    26: 'Hydrated Borates Containing Hydroxyl or Halogen',
    27: 'Compound Borates',
    28: 'Anhydrous Acid and Sulfates',
    29: 'Hydrated Acid and Sulfates',
    30: 'Anhydrous Sulfates Containing Hydroxyl or Halogen',
    31: 'Hydrated Sulfates Containing Hydroxyl or Halogen',
    32: 'Compound Sulfates',
    33: 'Selenates and Tellurates',
    34: 'Selenites, Tellurites and Sulfites',
    35: 'Anhydrous Chromates',
    36: 'Compound Chromates',
    37: 'Anhydrous Acid Phosphates, Arsenates, and Vanadates',
    38: 'Anhydrous Normal Phosphates, Arsenates, and Vanadates',
    39: 'Hydrated Acid Phosphates, Arsenates, and Vanadates',
    40: 'Hydrated Normal Phosphates, Arsenates, and Vanadates',
    41: 'Anhydrous Phosphates, Arsenates, and Vanadates containing Hydroxyl or Halogen',
    42: 'Hydrated Phosphates, Arsenates, and Vanadates containing Hydroxyl or Halogen',
    43: 'Compound Phosphates, Arsenates, and Vanadates',
    44: 'Antimonates',
    45: 'Acid and Normal Antimonates and Arsenites',
    46: 'Basic or Halogen-Containing Antimonites, Arsenites',
    47: 'Vanadium Oxysalts',
    48: 'Anhydrous Molybdates and Tungstates',
    49: 'Basic and Hydrated Molybdates and Tungstates',
    50: 'Salts of Organic Acids',
    51: 'Insular SiO4 Groups Only',
    52: 'Insular SiO4 Groups and O, OH, F, and H2O',
    53: 'Insular SiO4 Groups and Other Anions or Complex Cations',
    54: 'Borosilicates and Some Beryllosilicates',
    55: 'Si2O7 Groups, Generally with No Additional Anions',
    56: 'Si2O7 with Additional O, H, F and H2O',
    57: 'Insular Si3O10 and Larger Noncyclic Groups',
    58: 'Insular, Mixed, Single, and Larger Tetrahedral Groups',
    59: 'Three-Membered Rings',
    60: 'Four-Membered Rings',
    61: 'Six-Membered Rings',
    62: 'Eight-Membered Rings',
    63: 'Condensed Rings',
    64: 'Rings with Other Anions and Insular Silicate Groups',
    65: 'Single-Width Unbranched Chains W=1',
    66: 'Double-Width Unbranched Chains W=2',
    67: 'Unbranched Chains with W>2',
    68: 'Structures with Chains of More Than One Width',
    69: 'Chains with Side Branches or Loops',
    70: 'Column or Tube Structures',
    71: 'Sheets of Six-Membered Rings',
    72: 'Two-Dimensional Infinite Sheets with Other Than Six-Membered Rings',
    73: 'Condensed Tetrahedral Sheets',
    74: 'Modulated Layers',
    75: 'Si Tetrahedral Frameworks',
    76: 'Al-Si Frameworks',
    77: 'Zeolites',
    78: 'Unclassified Silicates'
}


dana_class_groups = {
    'Native Elements and Alloys': [1],
    'Sulfides and Related Compounds': [2,3],
    'Oxides': range(4, 9),
    'Halogenides': range(9, 13),
    'Carbonates': range(13, 18),
    'Nitrates': range(18, 21),
    'Iodates': range(21, 24),
    'Borates': range(24, 28),
    'Sulfates': range(28, 33),
    'Selenates and Tellurates; Selenites and Tellurites': [33,34],
    'Chromates': [35,36],
    'Phosphates, Arsenates, and Vanadates': range(37, 44),
    'Antimonates, Antimonites, and Arsenites': range(44, 47),
    'Vanadium Oxysalts': [47],
    'Molybdates and Tungstates': [48,49],
    'Organic Compounds': [50],
    'Nesosilicates: Insular SiO4': range(51, 55),
    'Sorosilicates: Isolated Tetrahedral Noncyclic Groups': range(55, 59),
    'Cyclosilicates': range(59, 65),
    'Inosilicates: Two-Dimensionally Infinite Silicate Units': range(65, 71),
    'Phyllosilicates': range(71, 75),
    'Tektosilicates': range(75, 78),
    'Unclassified': [78]
}
