import logging; logging.basicConfig(
    level=logging.INFO); pylogger = logging.getLogger(__name__)
import os
import pathlib
from aware import REPO_ROOT as path_to_root

import yaml
import hydra 
import re 

from aware.utils.modelsaver import ModelSaver, LEGACY_PACKAGES


def load_agent(timestamp, run_name="run", id=None, auto_infer_load_args=False, agent_file_starts="Agent", foldername=None,
               config_file="config.yaml", return_configs=False, suffix_numbering=True, device=None, 
               check_dirs=["checkpoints", "outputs"], 
               **kwargs):
  """
  Load an agent training defined at a specific timestep, and with a given run name. 
  Optionally, specify an id to load (None means load the most recently saved model).

  'timestamp' is either a path to a run folder, or a timestamp (YYYY-MM-DD_Hr-Mn-ss)
  which is searched for inside check_dirs.

  Kwargs specifies arguments for ModelSaver
  """

  # find if the run exists
  if os.path.isdir(timestamp):
    run_folder = timestamp
  else:
    run_folder = find_run(timestamp, check_dir=check_dirs,
                          error_if_not_unique=True)
  logging.info(f"Run found at {run_folder}")

  # load the run and its configs
  run_path = f"{run_folder}/{run_name}"
  modelloader = ModelSaver(run_path, **kwargs)
  agent_configs = yaml.safe_load(pathlib.Path(f"{run_path}/{config_file}").read_text())

  if auto_infer_load_args:
    auto_agent_file_start, auto_compression = auto_infer_model_load_args(run_path)
    if auto_agent_file_start is not None and auto_compression is not None:
      agent_file_starts = auto_agent_file_start
      if auto_compression == 'pkl':
        modelloader.use_compression = False
        modelloader.uncompressed_extension = '.pkl'
      else:
        modelloader.use_compression = True
        modelloader.compressor = auto_compression
    else:
      logging.warning(f"Failed to inferred agent file start and extension from {run_path}, instead using default values!")

  agent_save_state = modelloader.load(filenamestarts=agent_file_starts, foldername=foldername,
                                      id=id, suffix_numbering=suffix_numbering)
  # fix for backwards compatibility: runs saved before the code was packaged as 'aware'
  agent_configs["model"] = remap_legacy_targets(agent_configs["model"])

  # enable loading in an 'eval_mode'
  if agent_configs["model"]["_target_"] == "aware.agents.latent_predictor.Agent_Latent_Estimator":
    agent_configs["model"]["eval_mode"] = True

  if device is not None and "device" in agent_save_state:
    agent_save_state["device"] = device

  agent = hydra.utils.instantiate(agent_configs["model"])
  # load the checkpoint into the agent, and get the final policy
  agent.load_save_state(agent_save_state)

  logging.info(f"Finished loading '{agent_configs['model']['_target_']}' " 
               f"from timestamp '{timestamp}', and id={id}")
  
  if return_configs:
    return agent, agent_configs
  else:
    return agent

def remap_legacy_targets(cfg):
  """
  Recursively prefix hydra '_target_' entries from legacy top-level modules with 'aware.'
  """
  if isinstance(cfg, dict):
    return {k: (f"aware.{v}" if k == "_target_" and isinstance(v, str)
                and v.split(".")[0] in LEGACY_PACKAGES else remap_legacy_targets(v))
            for k, v in cfg.items()}
  if isinstance(cfg, list):
    return [remap_legacy_targets(x) for x in cfg]
  return cfg

def find_run(timestamp, check_dir="outputs", error_if_not_unique=True):
  """
  Returns the path to a run given a timestamp. This should have the format:

    -> YYYY-MM-DD_Hr-Mn-ss
    eg 2025-03-10_12-00-00

  The seconds and minutes are optional, if there is only one run that starts
  with the hour/min.
  """
  if isinstance(check_dir, str):
    check_dir = [check_dir]

  for dir in check_dir:

    if os.path.isabs(dir):
      loadpath = dir
    else:
      loadpath = f"{path_to_root}/{dir}"
    day = timestamp.split("_")[0]
    time = timestamp.split("_")[1]

    path_to_day_folder = f"{loadpath}/{day}"

    final_path = None

    if os.path.exists(path_to_day_folder):

      all_runs = [x for x in os.listdir(path_to_day_folder) 
                  if x.startswith(time)]
      
      if len(all_runs) == 1:
        final_path = f"{path_to_day_folder}/{all_runs[0]}"
        pylogger.info(f"find_run(): run found from timestamp '{timestamp}', at path: {final_path}")
        break
      elif len(all_runs) > 1:
        pylogger.warning(f"find_run() warning: timestamp given = '{timestamp}'. "
                            f"Multiple run candidates with time starting '{time}' found.\n"
                            f"Possible runs are: {all_runs}")
        final_path = []
        for i in range(len(all_runs)):
          this_path = f"{path_to_day_folder}/{all_runs[i]}"
          final_path.append(this_path)
        break
      else:
        pylogger.info(f"find_run(): timestamp given = '{timestamp}'. "
                      f"No run with time starting '{time}' found at path: {path_to_day_folder}")
    else:
      pylogger.info(f"find_run(): timestamp given = '{timestamp}'. "
                    f"Run date folder '{day}' not found at path: {path_to_day_folder}")
      
  if error_if_not_unique:
    if final_path == None:
      raise RuntimeError(f"find_run() could not find any run, given timestamp={timestamp}"
                         f", and check_dir={check_dir}")
    elif isinstance(final_path, list):
      raise RuntimeError(f"find_run() found multiple runs, given timestamp={timestamp}"
                         f", and check_dir={check_dir}.\n"
                         f"Runs were at path: {path_to_day_folder}\n"
                         f"Runs found were: {final_path}")
  
  return final_path

def auto_infer_model_load_args(path):

    files = os.listdir(path)
    valid_formats = {'.pkl', '.lz4', '.pbz2'}
    found_formats = set()
    model_ckpts = []

    for filename in files:
      # skip configs
      if filename.endswith('.yaml') or filename.endswith('.jsonl'):
        continue 
      
      # skip sub-directories (like /media)
      full_path = os.path.join(path, filename)
      if os.path.isdir(full_path):
        continue

      file_ext = os.path.splitext(filename)[1]
      if file_ext in valid_formats:
        found_formats.add(file_ext)
        model_ckpts.append(filename)
      else:
        return (None, None)
    
    if len(model_ckpts) == 0 or len(found_formats) > 1:
      return (None, None)
    
    found_format = found_formats.pop()
    pattern = re.compile(rf"(.+)_(\d+){re.escape(found_format)}$")
    ckpt_matches = all([pattern.match(f) for f in model_ckpts])
    if not ckpt_matches:
      return (None, None)
    else:
      filenamestarts = "_".join(model_ckpts[0].split("_")[:-1])
      compression_format = model_ckpts[0].split(".")[-1]
      return (filenamestarts, compression_format)
