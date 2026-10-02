import collections
import concurrent.futures
import json
import os
import re
import yaml

import io
import logging

import numpy as np
import wandb
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from elements import path, printing, timer, when, counter

def init_logging(cfg):
  """
  Convenience function to create a save directory and start logging using default
  settings with weights and biases.
  """

  if os.getenv("LOGGING", "true").lower() == "true":

    cfg = create_savedir(cfg)

    logging.info("logger.py -> init_logging() called. Creating a logger with default"
                " settings.\n"
                f" -> savedir = {cfg.savedir}\n"
                f" -> log_to_wandb = {cfg.logging.log_to_wandb}")
    
    # os.mkdir("wandb")
    # path = f"{os.getcwd()}/wandb"
    # print(f"path: {path}")
    # st = os.stat(path)
    # print(oct(st.st_mode))
    # print(os.getuid(), os.geteuid())

    # print(f"access: {os.access(path, os.W_OK)}")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

    # initialise weights and biases
    if cfg.logging.log_to_wandb:
      wandb_instance = wandb.init(config=cfg_dict, mode=os.getenv("WANDB_MODE", "online"), **cfg_dict['wandb'])
    else:
      wandb_instance = None

    # create the local logger
    logger = ProjectLogger(wandb_instance=wandb_instance, project_config=cfg_dict,
                          **cfg.logging)
 
  else:

    logging.info("logger.py -> init_logging() called. Creating debug logger.\n" \
                f" -> no save directory created\n"
                f" -> no wandb logging"
                f" -> terminal logging still enabled")
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

    logger = ProjectLogger(10, log_to_wandb=False, log_to_JSON=False)

  return logger, cfg

def create_savedir(cfg):
  """
  Create a new folder to save the run outputs in. If a folder with the same name
  exists, it adds progressively increasing integers on the end (_1, _2, etc).
  This function is safe from race conditions.
  """

  folderpath = cfg.savedir
  # Keep the original path name for logging purposes
  original_path = folderpath

  # Handle the case where the folder already exists
  if os.path.exists(folderpath):
    i = 1
    # Find a new folder name that does not exist
    while os.path.exists(f"{original_path}_{i}"):
      i += 1
    
    # Update the folderpath to the new, unique name
    folderpath = f"{original_path}_{i}"
    
    logging.warning(
        f"logger.py -> create_savedir() warning: savedir requested '{original_path}' "
        f"already existed, changing savedir to '{folderpath}'")

    # Update the modified save location in the config
    cfg.savedir = folderpath
  
  # Create the directory. exist_ok=True prevents errors if another process
  # creates the directory between the check and this call.
  os.makedirs(folderpath, exist_ok=True)

  logging.info(
      f"logger.py -> create_savedir(): savedir for this run created at: '{cfg.savedir}'")

  return cfg

def create_table(widths, types=None, float_fmt=".1f", delim=" | "):
  """
  Creates a table where the first field is a name and is formatted as a string,
  and all other fields are formatted as floats.
  """
  header_str = """"""
  row_str = """"""

  for i in range(len(widths)): 
    this_delim = "" if i == 0 else delim
    header_str +=  "{0}{{{1}:<{2}}}".format(this_delim, i, widths[i])
    if types is None or types[i] in (str, int):
      row_str += "{0}{{{1}:<{2}}}".format(this_delim, i, widths[i])
    elif types[i] == float:
      row_str += "{0}{{{1}:<{2}{3}}}".format(this_delim, i, widths[i], float_fmt)
  header_str += "\n"
  row_str += "\n"

  return header_str, row_str

class LoggingLevel:
  
  def __init__(self, level):
    """
    Class to provide a 'with' statemtn for python logging, so use as:

    from aware.utils.logger import LoggingLevel

    with LoggingLevel(logging.WARNING):
      ... # do something, and only messages at WARNING or above are printed
    """
    self.new_level = level
    self.original_level = None

  def __enter__(self):
    # Store the current logging level
    self.original_level = logging.getLogger().level
    # Set the new logging level
    logging.getLogger().setLevel(self.new_level)

  def __exit__(self, exc_type, exc_val, exc_tb):
    # Restore the original logging level when exiting the 'with' block
    logging.getLogger().setLevel(self.original_level)

class Logger:

  def __init__(self, step, outputs, multiplier=1):
    assert outputs, 'Provide a list of logger outputs.'
    self.step = step
    self.outputs = outputs
    self.multiplier = multiplier
    self._last_step = None
    self._last_time = None
    self._metrics = []

  @timer.section('logger_add')
  def add(self, mapping=None, prefix=None, step=None):

    if step is None:
      step = int(self.step) * self.multiplier
    else:
      step = int(step) * self.multiplier

    mapping = dict(mapping)
    # print('logger add:', len(mapping))
    assert len(mapping) <= 1000, list(mapping.keys())
    for key in mapping.keys():
      assert len(key) <= 200, (len(key), key[:200] + '...')
    for name, value in mapping.items():
      name = f'{prefix}/{name}' if prefix else name
      if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, str):
        value = str(value)
      if not isinstance(value, str):
        value = np.asarray(value)
        if len(value.shape) not in (0, 1, 2, 3, 4):
          raise ValueError(
              f"Shape {value.shape} for name '{name}' cannot be "
              "interpreted as scalar, vector, image, or video.")
      self._metrics.append((step, name, value))

  def scalar(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) == 0, value.shape
    self.add({name: value})

  def vector(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) == 1, value.shape
    self.add({name: value})

  def image(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) in (2, 3), value.shape
    self.add({name: value})

  def video(self, name, value):
    value = np.asarray(value)
    # assert len(value.shape) == 4, value.shape
    self.add({name: value})

  def text(self, name, value):
    assert isinstance(value, str), (type(value), str(value)[:100])
    self.add({name: value})

  @timer.section('logger_write')
  def write(self):
    if not self._metrics:
      return
    for output in self.outputs:
      with timer.section(type(output).__name__):
        output(tuple(self._metrics))
    self._metrics.clear()

  def close(self):
    self.write()
    for output in self.outputs:
      if hasattr(output, 'wait'):
        try:
          output.wait()
        except Exception as e:
          print(f'Error waiting on output: {e}')

class ProjectLogger(Logger):

  def __init__(self, log_rate, log_to_wandb=True, log_to_terminal=True, 
               log_to_JSON=True, wandb_instance=None, main_logdir="./",
               json_filename="metrics.jsonl", media_logdir="media",
               media_log_rate=None, debug_log_level=1, video_frames=200,
               project_config=None, save_configs_as="config.yaml"):
    """
    Logger that has multiple output streams:
      - terminal
      - weights and biases (wandb) [requires wandb_instance]
      - JSON files [requires logdir]

    Output streams are logged to at a specified log rate.

    Use this logger as follows:

    # initialise
    logger = ProjectLogger(...)

    # log data after every episode/training step
    logger.scalar('foo', 42)
    logger.scalar('foo', 43)
    logger.scalar('foo', 44)
    logger.vector('vector', np.zeros(100))
    logger.image('image', np.zeros((800, 600, 3, np.uint8)))
    logger.video('video', np.zeros((100, 64, 64, 3, np.uint8)))
    logger.log_step()

    For more see: https://github.com/danijar/elements
    """

    self.main_logdir = main_logdir
    self.log_rate = log_rate
    self.log_when = when.Every(log_rate)
    step = counter.Counter()
    self.episodes_done = int(step)
    self.log_to_wandb = log_to_wandb
    self.log_to_JSON = log_to_JSON
    self.log_to_terminaml = log_to_terminal
    self.media_logdir = media_logdir
    self.media_log_rate = media_log_rate
    self.debug_log_level = debug_log_level
    self.video_frames = video_frames
    self.wandb_instance = wandb_instance

    # create the main logdir
    if not os.path.exists(main_logdir):
      os.makedirs(main_logdir)

    outputs = []

    if log_to_terminal:
      outputs.append(TerminalOutput())

    if log_to_wandb:
      if wandb_instance == None:
        print("ProjectLogger.__init__() error: log_to_wandb=True, but no wandb instance given. wandb logging disabled")
      else:
        outputs.append(WandBOutput(wandb_instance))

    if log_to_JSON:
      outputs.append(JSONLOutput(main_logdir, json_filename))

    if project_config != None:
      if not isinstance(project_config, dict):
        project_config = OmegaConf.to_container(project_config, resolve=True)
      with open(f"{main_logdir}/{save_configs_as}", "w") as outfile:
        yaml.dump(project_config, outfile)

    if len(outputs) == 0:
      print("ProjectLogger.__init__() error: log_to_wandb, log_to_JSON, log_to_terminal are ALL False, nothing will be logged")

    super().__init__(step, outputs)

  def log_step(self, print_string=None):
    """
    Triggers a write action at the specified log rate, and increments the count
    of episodes done. Optionally print a string as well.
    """
    if self.log_when(self.episodes_done):
      self.write()
      if print_string is not None: print(print_string)
    self.step.increment()
    self.episodes_done = int(self.step)

  def log_dict(self, dict_or_value, key_str=""):
    """
    Log the leaves of a dictionary as scalar values.

    For example, this dictionary input:

    example = {
      "Train" : {
        "Reward" : 0.43,
        "Length" : 125,
      }
      "Loss" : 1.4e-1,
    }

    would expand to:

    self.scalar("Train/Reward", 0.43)
    self.scalar("Train/Length", 125)
    self.scalar("Loss", 1.4e-1)
    """
    if dict_or_value is {}:
      return
    elif isinstance(dict_or_value, dict):
      for key, value in dict_or_value.items():
        if key_str == "": new_str = f"{key}"
        else: new_str = f"{key_str}/{key}"
        self.log_dict(value, new_str)
    else:
      dict_or_value = np.asarray(dict_or_value)
      self.add({key_str: dict_or_value})

class AsyncOutput:

  def __init__(self, callback, parallel=True):
    self._callback = callback
    self._parallel = parallel
    if parallel:
      name = type(self).__name__
      self._worker = concurrent.futures.ThreadPoolExecutor(
          1, f'logger_{name}_async')
      self._future = None

  def wait(self):
    if self._parallel and self._future:
      concurrent.futures.wait([self._future])

  def __call__(self, summaries):
    if self._parallel:
      self._future and self._future.result()
      self._future = self._worker.submit(self._callback, summaries)
    else:
      self._callback(summaries)

class TerminalOutput:

  def __init__(self, pattern=r'.*', name=None, limit=50):
    self._pattern = (pattern != r'.*') and re.compile(pattern)
    self._name = name
    self._limit = limit

  def __call__(self, summaries):

    step = max(s for s, _, _, in summaries)
    scalars = {
        k: float(v) for _, k, v in summaries
        if isinstance(v, np.ndarray) and len(v.shape) == 0}
    if self._pattern:
      scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
    else:
      truncated = 0
      if len(scalars) > self._limit:
        truncated = len(scalars) - self._limit
        scalars = dict(list(scalars.items())[:self._limit])
    formatted = {k: self._format_value(v) for k, v in scalars.items()}
    if self._name:
      header = f'{"-" * 20}[{self._name} Step {step:_}]{"-" * 20}'
    else:
      header = f'{"-" * 20}[Step {step:_}]{"-" * 20}'
    content = ''
    if self._pattern:
      content += f"Metrics filtered by: '{self._pattern.pattern}'"
    elif truncated:
      content += f'{truncated} metrics truncated, filter to see specific keys.'
    content += '\n'
    if formatted:
      content += ' / '.join(f'{k} {v}' for k, v in formatted.items())
    else:
      content += 'No metrics.'
    printing.print_(f'\n{header}\n{content}\n', flush=True)

  def _format_value(self, value):
    value = float(value)
    if value == 0:
      return '0'
    elif 0.01 < abs(value) < 10000:
      value = f'{value:.2f}'
      value = value.rstrip('0')
      value = value.rstrip('0')
      value = value.rstrip('.')
      return value
    else:
      value = f'{value:.1e}'
      value = value.replace('.0e', 'e')
      value = value.replace('+0', '')
      value = value.replace('+', '')
      value = value.replace('-0', '-')
    return value

class JSONLOutput(AsyncOutput):

  def __init__(
          self, logdir, filename='metrics.jsonl', pattern=r'.*',
          strings=False, parallel=True):
    super().__init__(self._write, parallel)
    self._pattern = re.compile(pattern)
    self._strings = strings
    self.path_created = False
    self.given_filename = filename
    self.given_logdir = logdir

  @timer.section('jsonl')
  def _write(self, summaries):
    if not self.path_created:
      logdir = path.Path(self.given_logdir)
      logdir.mkdir()
      self._filename = logdir / self.given_filename
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      if not self._pattern.search(name):
        continue
      if isinstance(value, str) and self._strings:
        bystep[step][name] = value
      if isinstance(value, np.ndarray) and len(value.shape) == 0:
        bystep[step][name] = float(value)
    lines = ''.join([
        json.dumps({'step': step, **scalars}) + '\n'
        for step, scalars in bystep.items()])
    printing.print_(f'Writing metrics: {self._filename}')
    with self._filename.open('a') as f:
      f.write(lines)

class WandBOutput:

  def __init__(self, wandb_instance=None, pattern=r'.*', **kwargs):
    self._pattern = re.compile(pattern)
    import wandb
    self._wandb = wandb
    # if wandb_instance is None:
    #   import wandb
    #   wandb.init(**kwargs)
    #   self._wandb = wandb
    # else: self._wandb = wandb_instance

  def __call__(self, summaries):
    bystep = collections.defaultdict(dict)
    wandb = self._wandb
    for step, name, value in summaries:
      if not self._pattern.search(name):
        continue
      if isinstance(value, str):
        bystep[step][name] = value
      elif len(value.shape) == 0:
        bystep[step][name] = float(value)
      elif len(value.shape) == 1:
        # Check for mean/std pairs for line plots with shading
        if name.endswith("/mean"):
          base_name = name[:-5]
          std_name = base_name + "/std"
          # Store for deferred plotting
          if "line_series" not in bystep[step]:
            bystep[step]["line_series"] = {}
          bystep[step]["line_series"][base_name] = {"mean": value}
        elif name.endswith("/std"):
          base_name = name[:-4]
          if "line_series" not in bystep[step]:
            bystep[step]["line_series"] = {}
          if base_name not in bystep[step]["line_series"]:
            bystep[step]["line_series"][base_name] = {}
          bystep[step]["line_series"][base_name]["std"] = value
        else:
          bystep[step][name] = wandb.Histogram(value)
      elif len(value.shape) in (2, 3):
        value = value[..., None] if len(value.shape) == 2 else value
        assert value.shape[2] in [1, 3, 4], value.shape
        if value.dtype != np.uint8:
          value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
        bystep[step][name] = wandb.Image(value)
      elif len(value.shape) == 4:
        assert value.shape[3] in [1, 3, 4], value.shape
        value = np.transpose(value, [0, 3, 1, 2])
        if ~np.all((value >= 0) & (value <= 255)):
          value = 255 * np.clip(value, 0, 1)
        if value.dtype != np.uint8:
          value = value.astype(np.uint8)
        bystep[step][name] = wandb.Video(value, fps=50)

    # After collecting, process any mean/std line series for line plots with shading
    for step, metrics in list(bystep.items()):
      if "line_series" in metrics:
        for name, data in metrics["line_series"].items():
          mean = data.get("mean")
          std = data.get("std")
          if mean is not None and std is not None:
            xs = list(range(len(mean)))
            table = wandb.Table(columns=["x", "mean", "upper", "lower"])
            for i in range(len(mean)):
              table.add_data(xs[i], float(mean[i]), float(mean[i] + std[i]), float(mean[i] - std[i]))
            plot = wandb.plot.line_series(
                xs=table.get_column("x"),
                ys=[table.get_column("mean")],
                keys=["mean"],
                title=name,
                xname="Timestep",
            )
            metrics[name] = plot
        del metrics["line_series"]

    for step, metrics in bystep.items():
      self._wandb.log(metrics, step=step)

def fig_to_img(fig):
  with io.BytesIO() as buf:
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    img = plt.imread(buf, format='png')[..., :3]
  if img.dtype != np.uint8:
    img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
  plt.close(fig)
  return img
