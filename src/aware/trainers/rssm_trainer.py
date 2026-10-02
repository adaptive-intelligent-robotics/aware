import logging; pylogger = logging.getLogger(__name__)
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import elements
from typing import Dict
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from aware import REPO_ROOT as path_to_root
from aware.utils.modelsaver import ModelSaver
from aware.agents.prediction_discriminator import Agent_Prediction_Discriminator
from aware.agents.latent_discriminator import Agent_Latent_Discriminator
from aware.agents.double_discriminator import Agent_Double_Discriminator
from aware.agents.dataloader import Lz4MultiFileDataset, JaxPrefetcher
from aware.agents.latent_predictor import NoiseGenerator
from aware.evaluation import anomaly

class RSSMTrainer:

  def __init__(self,
               model,
               logger,
               dataset_path,
               batch_size,
               batch_length,
               steps,
               num_epochs,
               eval_every,
               save_dir,
               context_length=None,
               noise_generator_cfg=None,
               use_noise_generator=False,
               num_workers=2,
               prefetch_factor=2,
               debug=False,
               skip=None):

    # setup core training components
    self.model = model
    self.enable_roa = self.model.enable_roa
    self.logger = logger
    self.debug = debug
    self.skip = [] if skip is None else skip
    if any([x not in ['discriminator_eval', 'logging'] for x in self.skip]):
      raise ValueError("Invalid entries in skip list! Please only enter 'discriminator_eval' and/or 'logging'.")
    
    if debug:
      pylogger.setLevel(logging.DEBUG)
    else:
      pylogger.setLevel(logging.INFO)

    pylogger.info(f"DEBUG = {self.debug}, SKIPPING COMPONENTS: {self.skip}")

    # training hyper params 
    self.num_steps = steps
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.batch_length = batch_length
    self.context_length = self.model.num_hist_timesteps_to_use if context_length is None else context_length
    self.eval_every = eval_every # this is now time steps
  
    # containers for logging    
    self.train_agg = elements.Agg()
    self.dataloader_agg = elements.Agg()
    self.count_agg = elements.Agg() 

    # model saving 
    self.model_saver = None
    self.save_dir = save_dir
    if save_dir: 
      self.model_saver = ModelSaver(save_dir, use_compression=False)    

    # offline data loading from a generated dataset
    self.num_envs = 1
    self.use_noise_generator = use_noise_generator
    self.num_workers = num_workers
    self.prefetch_factor = prefetch_factor

    # setup noise generator
    if self.use_noise_generator:
      noise_generator_dict = make_noise_generators(noise_generator_cfg)
    else:
      noise_generator_dict = {}

    dataset_path = os.path.join(path_to_root, dataset_path)
    ALL_FILES = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".lz4")]

    dataset = Lz4MultiFileDataset(file_paths=ALL_FILES,
                                  batch_size=self.batch_size,
                                  batch_length=self.batch_length+self.context_length,
                                  include_priv_info=self.enable_roa,
                                  **noise_generator_dict
                                  )

    loader = DataLoader(dataset, 
                        batch_size=None, # DISABLES AUTO BATCHING
                        batch_sampler=None, # DISABLES AUTO BATCHING
                        collate_fn=numpy_collate,
                        num_workers=self.num_workers,
                        prefetch_factor=self.prefetch_factor,
                        persistent_workers=True,
                        )
    
    if self.debug:
      # load an example file
      test_data = dataset.load_and_parse_file(ALL_FILES[0])
      pylogger.info(f"Test data shape: {test_data.shape}")

    device = jax.devices()[0]
    self.train_loader = JaxPrefetcher(loader, device)
    logging.info(f"Dataset loader initialized with prefetch_factor {self.prefetch_factor} and num_workers {self.num_workers}")
    
    # load the released real-world trajectories for evaluation
    self.eval_set = {name: anomaly.load_dataset(name) for name in anomaly.DATASETS}

    # debug printing 
    pylogger.info(f"MODEL STATUS:\n"
                 f"model roa = {self.enable_roa}\n"
                 f"model obs_dim = {self.model.obs_dim}\n"
                 f"model decoder loss = {self.model.dec.output_loss}\n"
                 f"model target_obs = {self.model.target_obs}\n"
                 f"model seed = {self.model.seed}\n"
                 f"model prior dropout = {self.model.dyn.prior_dropout_p}\n"
                 f"model posterior dropout = {self.model.dyn.posterior_dropout_p}\n"
                 f"model latent concat = {self.model.latent_concat}\n"
                 f"\n"
                 f"TRAINER STATUS:\n"
                 f"using noise generator = {self.use_noise_generator}")

  ## Training ##

  def train(self):
    
    self.init_carry, self.init_act = self.model.init_carry(self.batch_size)
    step = 0        

    for epoch in range(self.num_epochs):

      if step >= self.num_steps:
        break

      train_iter = iter(self.train_loader)
      while step < self.num_steps:
        t0_wait = time.perf_counter()
        try:
          batch_dict = next(train_iter)
        except StopIteration:
          break
        t1_wait= time.perf_counter()
        wait_time = t1_wait - t0_wait 

        # remove the extra features (noise information from the batch)
        extras = batch_dict.pop('extras', None)
  
        # 1. generate context by forward pass through RSSM and add to batch
        t1 = time.time()
        obs = {k: batch_dict[k] for k in self.model.obs_space} # obs_space + priv_vector_space
        prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
        prevact = {k: prepend(self.init_act[k], batch_dict[k]) for k in self.model.act_space if k != 'priv_latent'}
        assert all((v.shape[1] == (self.batch_length + self.context_length) for v in obs.values()))
        assert all((v.shape[1] == (self.batch_length + self.context_length) for v in prevact.values()))
        warmup_length = self.context_length
        
        # 2. warmup the RSSM
        if warmup_length != 0:
          obs_hist = {k: v[:, :warmup_length] for k, v in obs.items()}
          act_hist = {k: v[:, :warmup_length] for k, v in prevact.items()}
          if self.enable_roa:
            assert 'priv_vector' in obs_hist
            priv_latent = self.model.roa_encode(obs_hist) # B, T, D
            priv_latent = jax.lax.stop_gradient(priv_latent) # this is important -> update I don't think it is 
            if self.model.latent_concat == "obs":
              obs_hist['priv_latent'] = priv_latent
            elif self.model.latent_concat == "act":
              act_hist['priv_latent'] = priv_latent
          carry, _, _ = self.model.forward(self.init_carry, obs_hist, act_hist)
        else:
          carry = self.init_carry

        # 3. extract train data from warmup data
        train_obs = {k: v[:, self.context_length:] for k, v in obs.items()}
        init_act = {k: v[:, self.context_length:] for k, v in prevact.items()}
        extras = {k: v[:, self.context_length:] for k, v in extras.items()}
        assert all((v.shape[1] == self.batch_length for v in train_obs.values()))
        assert all((v.shape[1] == self.batch_length for v in init_act.values()))
        assert all((v.shape[1] == self.batch_length for v in extras.values()))

        data = {**train_obs,
                **init_act,
                **extras}

        data_shapes = {k: v.shape for k, v in data.items()}
        load_context_time = time.time() - t1

        # 4. train
        t0_train = time.perf_counter()
        _, mets = self.model.train(carry, self.init_act, data, int(step))
        if self.debug:
          jax.tree_util.tree_map(lambda x: x.block_until_ready(), mets)
        t1_train = time.perf_counter()
        train_time = t1_train - t0_train    

        # 5. update metrics
        deter_reset = mets.pop('deter_resets', 0)
        stoch_resets = mets.pop('stoch_resets', 0)
        self.count_agg.add('step', int(step), agg='last')
        self.count_agg.add('deter_resets', deter_reset, agg='sum')
        self.count_agg.add('stoch_resets', stoch_resets, agg='sum')
        self.count_agg.add('warmup_length', warmup_length, agg='mean')
        self.train_agg.add(mets, prefix='train')
        self.dataloader_agg.add('train_time', train_time, agg='mean')
        self.dataloader_agg.add('load_context_time', load_context_time, agg='mean')

        # if step % 10 == 0:
        #   print(f"Step: {step} | Data Wait: {wait_time*1000:.3f}ms | Train Time: {train_time*1000:.3f}ms")

        # 6. evaluation + logging + saving
        if step % self.eval_every == 0:
          pylogger.debug("running evaluation")
          if 'discriminator_eval' not in self.skip:
            d_metrics = self.run_discriminator_eval(int(step))
            self.logger.add(d_metrics, prefix='eval/discriminator') 

          if 'logging' not in self.skip:
            self.logger.add(self.dataloader_agg.result(), prefix='data_loader')
            self.logger.add({'epoch': epoch}, prefix='counters')
              
            self.logger.add(self.train_agg.result())
            self.logger.add(self.count_agg.result(reset=False), prefix='counters')
            self.logger.write()
            self.save_model(int(step))

        step += 1
        self.logger.step = step

   
  ## Evaluation ##

  def run_discriminator_eval(self, step):
    """
    Evaluate anomaly detection on the released real-world trajectories
    """

    pylogger.info(f"Running a discrimination test")

    # wrap the agent into discriminators
    disc_latent = Agent_Latent_Discriminator(agent=self.model, dummy_latents=not self.enable_roa)
    disc_pred = Agent_Prediction_Discriminator(agent=self.model)
    disc_double = Agent_Double_Discriminator(latent_discrim=disc_latent, 
                                             pred_discrim=disc_pred)

    test_logs = {}
    for case, trajectory in self.eval_set.items():
      eval_dict = anomaly.evaluate(disc_double, trajectory, name=f"Model step={step}",
                                   traj_label=case)
      for key in eval_dict:
        test_logs = test_logs | {
          f"{case}_{key}_f1" : eval_dict[key]["f1"],
          f"{case}_{key}_average_precision": eval_dict[key]["average_precision"],
          f"{case}_{key}_auroc": eval_dict[key]["auroc"],
          f"{case}_{key}_hz" : eval_dict[key]["approx_frequency"],
          f"{case}_{key}_plot" : eval_dict[key].get("plot", None),
        }

    # drop None entries
    test_logs = {k:v for k, v in test_logs.items() if v is not None} 

    return test_logs

  ## Model Saving ##

  def save_model(self, step):
    if self.model_saver:
      # only saving the model weights and not the opt values  
      weights = jax.device_get(self.model.model_params)
      self.model_saver.save(name="rssm", pyobj=weights, force_suffix=step)
      pylogger.info("Saved!")
    else:
      pylogger.info("Warning! Save called but no model_saver exists! State was not saved")


def make_noise_generators(noise_generator_cfg):

      assert isinstance(noise_generator_cfg, (Dict, DictConfig)), (f"Attempted to use noise generator but a valid cfg type was not given"
                                                                    f"(given type {type(noise_generator_cfg)})")
      # extract the noise_stds to use
      std_qpos = np.array(noise_generator_cfg['noise_std_SI_qpos'])
      std_qvel = np.array(noise_generator_cfg['noise_std_SI_qvel'])
      std_action = np.array(noise_generator_cfg['noise_std_SI_action'])
      std_priv_info = np.array(noise_generator_cfg['noise_std_SI_priv_info'])
      
      pylogger.info(f"Loaded noise values:\n"
                    f"std_qpos: {std_qpos}\n"
                    f"std_qvel: {std_qvel}\n"
                    f"std_action: {std_action}\n"
                    f"std_priv_info: {std_priv_info}\n")

      noise_params = noise_generator_cfg["noise_params"]
      # create noise generators based on noise parameters
      if ("master_std_scaling" in noise_params and 
          noise_params["master_std_scaling"] is not None):
        master_scale = { "std_scaling" : noise_params["master_std_scaling"] }
      else: master_scale = {}
      if ("master_std_range" in noise_params and 
          noise_params["master_std_range"] is not None):
        master_range = { "std_range" : noise_params["master_std_range"] }
      else: master_range = {}

      noise_generator_dict= {
        "vector_noise_generator": None,
        "vector_noise_generator_args": {},
        "action_noise_generator": None,
        "action_noise_generator_args": {},
        "priv_info_noise_generator": None,
        "priv_info_noise_generator_args": {}
      }

      if noise_generator_cfg['noise_vector']:
        noise_generator_dict["vector_noise_generator"] = NoiseGenerator(**dict(noise_params["qpos_qvel"]) | master_scale | master_range)
        noise_generator_dict["vector_noise_generator_args"] = {'std_base': std_qpos,
                                                              'std_correlated': std_qvel,
                                                              'use_np':True}
      if noise_generator_cfg['noise_action']:
        noise_generator_dict["action_noise_generator"] = NoiseGenerator(**dict(noise_params["action"]) | master_scale | master_range)
        noise_generator_dict["action_noise_generator_args"] = {"std_dev": std_action}
      
      if noise_generator_cfg['noise_priv_info']:
        noise_generator_dict["priv_info_noise_generator"] = NoiseGenerator(**dict(noise_params["priv_info"]) | master_scale | master_range)
        noise_generator_dict["priv_info_noise_generator_args"] = {"std_dev":std_priv_info}
      
      return noise_generator_dict

def numpy_collate(batch):
  if isinstance(batch, list):
    return batch[0]
  return batch
