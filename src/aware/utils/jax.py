"""
JAX helper functions
"""

from jax._src.dlpack import to_dlpack, from_dlpack
import torch.utils.dlpack as tpack


# from: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/wrapper_torch.py
def jax_to_torch(tensor):
  tensor = to_dlpack(tensor)
  tensor = tpack.from_dlpack(tensor)
  return tensor

def torch_to_jax(tensor):
  tensor = tpack.to_dlpack(tensor)
  tensor = from_dlpack(tensor)
  return tensor

# from elements
def map(fn, *trees, isleaf=None):
  assert trees, 'Provide one or more nested Python structures'
  kw = dict(isleaf=isleaf)
  first = trees[0]
  try:
    assert all(isinstance(x, type(first)) for x in trees)
    if isleaf and isleaf(trees[0]):
      return fn(*trees)
    if isinstance(first, list):
      assert all(len(x) == len(first) for x in trees)
      return [map(
          fn, *[t[i] for t in trees], **kw) for i in range(len(first))]
    if isinstance(first, tuple):
      assert all(len(x) == len(first) for x in trees)
      return tuple([map(
          fn, *[t[i] for t in trees], **kw) for i in range(len(first))])
    if isinstance(first, dict):
      assert all(set(x.keys()) == set(first.keys()) for x in trees)
      return {k: map(fn, *[t[k] for t in trees], **kw) for k in first}
    if hasattr(first, 'keys') and hasattr(first, 'get'):
      assert all(set(x.keys()) == set(first.keys()) for x in trees)
      return type(first)(
          {k: map(fn, *[t[k] for t in trees], **kw) for k in first})
  except AssertionError:
    raise TypeError(printing.format_(trees))
  return fn(*trees)




