"""
ninjax 3.6.2, the version used for the paper results, has a stray print() in flatten() which
prints every parameter and optimizer state path when a model is created. This replaces it
with an identical copy without the print. Later ninjax versions remove the print, but also
change other internals, so the version is kept pinned.
"""
import re

import jax
import ninjax.ninjax


def flatten(tree):
  items, treedef = jax.tree_util.tree_flatten_with_path(tree)
  paths, values = zip(*items)
  def tostr(key):
    key = key.key if hasattr(key, 'key') else key
    key = re.sub(r'[^A-Za-z0-9-_/]+', '', str(key))
    return key
  spaths = [[tostr(x) for x in path] for path in paths]
  keys = ['/'.join(x for x in spath if x) for spath in spaths]
  treedef = (keys, treedef)
  if len(set(keys)) < len(keys):
    raise ValueError(
        'Cannot flatten PyTree to dict because paths are ambiguous '
        'after converting them to string keys.\n'
        'Paths: {paths}\nKeys: {keys}')
  items = dict(sorted(list(zip(keys, values)), key=lambda x: x[0]))
  return items, treedef


if ninjax.ninjax.__version__ == '3.6.2':
  ninjax.ninjax.flatten = flatten
