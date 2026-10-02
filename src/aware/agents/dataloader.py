import torch
import torch.multiprocessing as mp
import numpy as np
from typing import List
import jax
import jax.numpy as jnp
import numpy as np
import yaml
import einops
from omegaconf import OmegaConf
from queue import Queue
from threading import Thread

import logging; logging.basicConfig(
    level=logging.INFO); pylogger = logging.getLogger(__name__)

from aware.utils.modelsaver import ModelSaver, lz4_decompress_pickle
from aware.utils.jax import torch_to_jax

# Configure multiprocessing for CUDA
mp.set_start_method('spawn', force=True)

class DataBuffer:

  def __init__(self, num_envs, traj_len, data_shape, device="cuda",
               dataset_path=None, save_data=False, save_style="numpy", 
               save_device="cpu"):

    # save inputs
    self.device = device
    self.num_envs = num_envs
    self.traj_len = traj_len
    self.data_shape = data_shape

    # create buffers
    self.traj_step = 0
    self.data = torch.zeros((num_envs, traj_len, *data_shape), device=self.device)

    # for saving
    self.save_data = save_data
    self.dataset_path = dataset_path
    self.save_style = save_style
    self.save_device = save_device
    if save_style == "jax":
      self.save_as = "data_jax"
    elif save_style == "torch":
      self.save_as = "data_torch"
    else:
      if save_style != "numpy":
        pylogger.warning(
            f"DataBuffer save_style='{save_style}' not recognised, defaulting to numpy")
      self.save_as = "data_numpy"

    if self.save_data:
      self.modelsaver = ModelSaver(self.dataset_path)
      self.total_n = 0
      self.running_mean = 0
      self.running_std = 0

  def add(self, new_data):
    """
    Add an observation to the buffer
    """
    if self.traj_step >= self.traj_len:
      raise RuntimeError(
          f"DataBuffer error: traj_len={self.traj_len} has been exceeded")
    self.data[:, self.traj_step].copy_(new_data)
    self.traj_step += 1
    return (self.traj_step >= self.traj_len)

  def is_full(self):
    return (self.traj_step >= self.traj_len)

  def empty(self):
    self.traj_step = 0
    return

  def get(self):
    """
    Return the whole buffer
    """
    return self.data

  def save(self, names=None, safety_check=True):
    """
    Save the buffer into a compressed file. 
    """

    # compute averages and deviations across the batch
    n = float(self.data.shape[0])
    means = torch.mean(self.data, dim=(0, 1))
    stds = torch.std(self.data, dim=(0, 1), unbiased=False) # sample std, not population

    # ---- unstable safety check!! ---- #

    if safety_check:

      safety_inds = torch.arange(24, 31)
      safety_threshold = 1e3
      qacc_std = stds[safety_inds]

      if torch.any(qacc_std > safety_threshold):
        pylogger.warning(f"Unstable behaviour detected!\n"
                         f" -> qacc_inds hardcoded as: {safety_inds}\n"
                         f" -> safety_threshold = {safety_threshold}\n"
                         f" -> qacc_stds = {qacc_std}")
        return # do not save unstable data
      else:
        pylogger.info(f"Stable behaviour detected, saving data now\n"
                      f" -> qacc_inds hardcoded as: {safety_inds}\n"
                      f" -> safety_threshold = {safety_threshold}\n"
                      f" -> qacc_stds = {qacc_std}")

    # ---- unstable safety check!! ---- #

    # calculate totals so far
    old_running_mean = self.running_mean
    self.running_mean = (self.total_n * self.running_mean + \
                         n * means) / (self.total_n + n)
    self.running_std = torch.sqrt(
        ((self.total_n * (self.running_std**2 + old_running_mean**2) + \
          n * (stds**2 + means**2)) /
         (self.total_n + n) - self.running_mean**2)
        .clamp(min=0.) # Clamp to avoid negative values due to floating point inaccuracies
    )
    self.total_n += n

    if names is None:
      names = ["data" for i in range(n)]

    log_str = f"""Mean and standard deviation of data, shape = {self.data.shape}\n\n"""
    header_str = f"{'Name':<20} | {'Index':<5} | {'Mean':<8} | {'Stddev':<8}\n"
    row_str = "{0:<20} | {1:<5} | {2:<8.4f} | {3:<8.4f}\n"

    log_str += header_str
    for i in range(self.data.shape[-1]):
      log_str += row_str.format(names[i], i, self.running_mean[i], self.running_std[i])

    # print(log_str) # for debugging, put the logs into the terminal

    # convert to the desired save type
    to_save = self.data.detach().to(self.save_device) # move to device, should be cpu
    if self.save_style == "jax":
      to_save = jnp.array(to_save)
    elif self.save_style == "numpy":
      to_save = to_save.numpy(force=True)

    # save the data, alongside a textfile with the statistics
    self.modelsaver.save(self.save_as, pyobj=to_save)
    self.modelsaver.save("dataset_statistics", txtstr=log_str, txtonly=True)

  def save_configs(self, configs, save_configs_as="config.yaml"):
    """
    Save a provided config dictionary in the dataset folder
    """

    savepath = self.modelsaver.get_current_path()

    pylogger.info(f"Saving dataset configs at: {savepath}/{save_configs_as}")

    if not isinstance(configs, dict):
        configs = OmegaConf.to_container(configs, resolve=True, throw_on_missing=False)
    
    with open(f"{self.modelsaver.get_current_path()}/{save_configs_as}", "w") as outfile:
      yaml.dump(configs, outfile)

  def load(self, id=None):
    """
    Load data from a fixed dataset into the buffer.
    """

    max_ind = self.modelsaver.get_recent_file(return_int=True)
    original_id = id
    if id is not None and id > max_ind - 1:
      id = (id % max_ind) + 1

    pylogger.info(f"Preparing to load dataset file. Given id={original_id}, "
                  f"loading id={id} / {max_ind} in dataset.")

    self.data = self.modelsaver.load(self.save_as, id=id).to(self.device)
    self.num_envs = self.data.shape[0]
    self.traj_len = self.data.shape[1]
    self.data_shape = self.data.shape
    self.traj_step = self.traj_len # mark buffer as full

  def to(self, device):
    self.data = self.data.to(device)
    self.device = device

class Dataset(torch.utils.data.Dataset):

  def __init__(self, dataset_path, datafile_name="data_numpy", device="cuda"):
    """
    Dataset wrapper for smooth loading
    """

    # save inputs
    self.dataset_path = dataset_path
    self.datafile_name = datafile_name
    self.device = device

    # create loading apparatus
    self.modelsaver = ModelSaver(self.dataset_path)
    self.num_files = self.modelsaver.get_recent_file(name=datafile_name, return_int=True)

    if self.num_files is None:
      pylogger.error(f"No files found for dataset")
      raise RuntimeError(f"Dataset.__init__() error: "
                         f"no dataset files found at path={dataset_path}, "
                         f"matching name={datafile_name} (file must start like this)")
    
    self.files_to_load = np.arange(1, self.num_files + 1)

    pylogger.info(f"Created a dataset from path: {dataset_path}\n"
                  f" -> datafile_name = {datafile_name}\n"
                  f" -> num_files = {self.num_files}\n")

  def __len__(self):
    return self.num_files

  def __getitem__(self, index):
    """
    Get a file from the dataset
    """
    if self.num_files is None:
      raise RuntimeError(f"Dataset.__getitem__() error: "
                         f"No files found, self.num_files=None")
    if index > self.num_files:
      raise RuntimeError(f"Dataset.__getitem__() error: "
                         f"index={index} greater than num_files={self.num_files}")

    # select and load sample
    id = self.files_to_load[index]
    data = self.modelsaver.load(self.datafile_name, id=id)

    return data

class DataLoadWrapper:

  def __init__(self, 
               dataset_path, 
               datafile_name="data_numpy", 
               data_mode="torch",
               data_indexes=None,
               device="cuda",
               shuffle=True,
               num_workers=1,     
               prefetch_factor=2, 
               **dataloader_args):
    """
    Class that wraps the dataset and exposes a 'load' function to get the next
    data file. Automatically handles epochs and uses pytorch dataloader. Optionally
    pass 'data_indexes = np.array([0, 1, 4, 6, 7], dtype=int)' to only
    extract from the data those indexes in the last dimension (data[:, :, indexes])
    """

    self.dataset = Dataset(
        dataset_path=dataset_path,
        datafile_name=datafile_name,
        device=device,
    )

    assert data_mode in ["torch", "jax", "cpu"], f"data_mode={data_mode} not recognised"
    self.data_mode = data_mode
    self.device = device
    self.data_indexes = data_indexes

    self.epoch = 0
    self.dataloader = torch.utils.data.DataLoader(
        self.dataset,
        shuffle=shuffle,                  # randomise the order of file loading
        num_workers=num_workers,          # number of extra processes doing loading
        prefetch_factor=prefetch_factor,  # each worker loads n files in advance
        batch_size=None,                  # we do not need the loader to form up batches
        batch_sampler=None,               # together with batch_size=None, this disables batching
        persistent_workers=True,          # continue to prefetch after epoch ends
        pin_memory=True,                  # move memory to specified device
        pin_memory_device=device,         # move it to this device
        **dataloader_args,
    )

    # create the iterator across the dataset
    self._iterator = iter(self.dataloader)

  def _reset_iterator(self):
    """
    Create a new iterator and let the dataloader class handle the epoch
    """
    self._iterator = iter(self.dataloader)
    self.epoch += 1

  def load(self):
    """
    Return the next batch of data. Automatically handles end-of-epoch and reshuffling.

    Returns:
        A batch of data from the DataLoader.
    """
    def get():
      # get the next chunk of data from the loader
      next_data = next(self._iterator)
      # handle indexing to remove data, before moving to GPU
      if self.data_indexes is None:
        self.data_indexes = np.arange(next_data.shape[-1]) # take it all
      next_data = next_data[:, :, self.data_indexes]
      # convert the data into the desired tensor, on the GPU
      if self.data_mode == "torch":
        return next_data.to(self.device)
      elif self.data_mode == "jax":
        return jax.device_put(torch_to_jax(next(self._iterator)))
      elif self.data_mode == "cpu":
        return np.array(next(self._iterator))
      else:
        raise RuntimeError(f"DataLoadWrapper.load() error: "
                           f"self.data_mode={self.data_mode} not recognised")

    try:
      return get()
    except StopIteration:
      self._reset_iterator()
      return get()


class Lz4MultiFileDataset(torch.utils.data.IterableDataset):

    def __init__(self, 
                 file_paths:List[str], 
                 batch_length:int, 
                 batch_size:int, 
                 include_priv_info:bool=True,

                 vector_noise_generator=None,
                 vector_noise_generator_args={},
                 action_noise_generator=None,
                 action_noise_generator_args={},
                 priv_info_noise_generator=None,
                 priv_info_noise_generator_args={},

                 **kwargs
                 ):

        self.file_paths = file_paths
        self.batch_size = batch_size
        self.batch_length = batch_length

        self.include_priv_info = include_priv_info

        # generator instances
        self.vector_noise_generator = vector_noise_generator
        self.action_noise_generator = action_noise_generator
        self.priv_info_noise_generator = priv_info_noise_generator

        # static args used to call the noise generator
        self.vector_noise_generator_args = vector_noise_generator_args
        self.action_noise_generator_args = action_noise_generator_args
        self.priv_info_noise_generator_args = priv_info_noise_generator_args

        self.is_first_template = np.zeros((self.batch_size, self.batch_length), dtype=bool); self.is_first_template[:, 0] = True
        self.is_last_template = np.zeros((self.batch_size, self.batch_length), dtype=bool); self.is_last_template[:, -1] = True
        self.reset_template = np.zeros((self.batch_size, self.batch_length), dtype=bool); self.reset_template[:, 0] = True

        # logging
        logging.info(f"Init Lz4MultiFileDataset complete!")
        logging.info(f"vector_noise_generator loaded: {self.vector_noise_generator is not None}")
        logging.info(f"vector_noise_generator_args: {self.vector_noise_generator_args}")
        logging.info(f"action_noise_generator loaded: {self.action_noise_generator is not None}")
        logging.info(f"action_noise_generator_args: {self.action_noise_generator_args}")
        logging.info(f"priv_info_noise_generator loaded: {self.priv_info_noise_generator is not None}")
        logging.info(f"priv_info_noise_generator_args: {self.priv_info_noise_generator_args}")


    def load_and_parse_file(self, file_path):
        """
        Decompresses lz4 and loads numpy array.
        Adjust this logic depending on if you used np.save or raw bytes.
        """
        try:
            data = lz4_decompress_pickle(file_path)
            trim = (data.shape[1] // self.batch_length) * self.batch_length
            # trim data to nearest multiple of batch_length
            data = data[:, :trim]
            # reshape to N batch_length sequences
            num_seqences = data.shape[0] * (data.shape[1] // self.batch_length)
            data = data.reshape((num_seqences, self.batch_length, data.shape[-1]))
            # trim out excess data
            data = data[..., :24]
            return data


        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None

    def process_batch(self, batch):
        """
        Process raw numpy data into a batch for training on GPU. 

        """

        vector = batch[..., :14]
        vector_noise = self.vector_noise_generator.add_correlated_noise(vector, **self.vector_noise_generator_args)
        action = batch[..., 14:17]
        action_noise = self.action_noise_generator.add_noise_np(action, **self.action_noise_generator_args)

        batch_dict = {
          'vector': vector + vector_noise,
          'action': action + action_noise,
          'is_first': self.is_first_template.copy(),
          'is_last': self.is_last_template.copy(),
          'reset': self.reset_template.copy(),

          'extras': {
            'vector_noise': vector_noise,
          }
        }

        if self.include_priv_info:
          priv_vector = batch[..., 17:24]
          priv_vector_noise = self.priv_info_noise_generator.add_white_noise_np(priv_vector, **self.priv_info_noise_generator_args)
          priv_vector_stds = self.priv_info_noise_generator.last_std_used[:, 0] # B 1 D -> B D
          priv_vector_stds_repeat = einops.repeat(priv_vector_stds, 'B D -> B T D', T=priv_vector.shape[1])

          batch_dict['priv_vector'] = priv_vector + priv_vector_noise
          batch_dict['extras'] |= {'priv_vector_noise': priv_vector_noise,
                                   'priv_vector_stds': priv_vector_stds_repeat}
        
        return batch_dict
    
    def __iter__(self):
        # 1. Distribute files among workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_files = self.file_paths
        else:
            # Split list: Worker K gets files [K, K+N, K+2N...]
            # This ensures better randomization than chunking [0..10], [11..20]
            my_files = self.file_paths[worker_info.id :: worker_info.num_workers]

        # 2. Iterate through assigned files
        for fp in my_files:
            data = self.load_and_parse_file(fp)
            if data is None: continue

            num_samples = data.shape[0]
            indices = np.random.permutation(num_samples)

            # 3. Yield Batches from this file
            for i in range(0, num_samples, self.batch_size):

                if i + self.batch_size > num_samples:
                    continue
                
                batch_indices = indices[i : i + self.batch_size]
                batch = data[batch_indices]
                processed_batch = self.process_batch(batch)
                yield processed_batch
            
            # Explicitly delete to free RAM for the next file load
            del data

class JaxPrefetcher:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.queue = Queue(maxsize=5) # Buffer 5 batches on GPU
        self.stream = None
        self.thread = None

    def _worker(self):
        try:
            for batch in self.dataloader:
                # This transfer happens in the background!
                # We use jax.device_put to move numpy (CPU) -> JAX Array (GPU)
                sharded_batch = jax.device_put(batch, self.device)
                self.queue.put(sharded_batch)
            self.queue.put(None) # Sentinel
        except Exception as e:
            print(f"Prefetcher error: {e}")
            self.queue.put(None)

    def __iter__(self):
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
        return self

    def __next__(self):
        batch = self.queue.get()
        if batch is None:
            self.thread.join()
            raise StopIteration
        return batch