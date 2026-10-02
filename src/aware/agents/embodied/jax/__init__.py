from . import ninjax_patch  # must come first, silences a stray print in ninjax 3.6.2

from .heads import DictHead
from .heads import Head
from .heads import MLPHead

from .opt import Optimizer

from . import nets
from . import outs
from . import opt
