import time
import logging; pylogger = logging.getLogger(__name__)

from aware.agents.latent_discriminator import Agent_Latent_Discriminator
from aware.agents.prediction_discriminator import Agent_Prediction_Discriminator
from aware.agents.double_discriminator import Agent_Double_Discriminator
from aware.utils.modelsaver import ModelSaver
from aware.evaluation import anomaly

class TransformerTrainer():

  def __init__(self, 
               model,
               logger,
               num_episodes,
               test_freq,
               save_freq,
               savedir,
               enable_saving,
               use_evaluator=True,
               debug=False,
               ):
    """
    Class that trains the Transformer AWARE agent from a generated dataset
    """

    # handle inputs
    self.agent = model
    self.logger = logger
    self.num_episodes = num_episodes
    self.test_freq = test_freq
    self.save_freq = save_freq
    self.ms_log_level = 1 # log level for modelsaver
    self.use_evaulator = use_evaluator
    self.debug = debug

    # prepare saving
    if savedir[-1] != "/": savedir += "/"
    self.savedir = savedir
    self.enable_saving = enable_saving
    if self.enable_saving:
      self.modelsaver = ModelSaver(savedir, log_level=self.ms_log_level)

    # must be exposed by the agent
    self.num_env_steps_per_update = self.agent.num_env_steps_per_update

    # load the released real-world trajectories for evaluation
    if self.use_evaulator:
      self.eval_set = {name: anomaly.load_dataset(name) for name in anomaly.DATASETS}

    # prepare tracking
    self.tot_timesteps = 0
    self.tot_time = 0
    self.current_learning_iteration = 0

    pylogger.info(f"TransformerTrainer has finished initialisation:\n"
                  f" -> num_episodes = {num_episodes}\n"
                  f" -> savedir = {savedir}\n"
                  f" -> save_freq = {save_freq}\n"
                  f" -> test_freq = {test_freq}\n"
                  f" -> use_evaluator = {use_evaluator}\n")

  def train(self):
    """
    Run the training of the Transformer AWARE model.
    """

    tot_iter = self.current_learning_iteration + self.num_episodes
    self.start_learning_iteration = self.current_learning_iteration

    pylogger.info(f"TransformerTrainer starting training at episode "
                  f"{self.current_learning_iteration}, with target of {tot_iter} episodes.")

    # train, iterating up to the target number of episodes
    for i_episode in range(self.current_learning_iteration, tot_iter + 1):

      # ensure model is in training mode
      self.agent.model.train()

      # load data from fixed dataset
      t0 = time.process_time()
      self.agent.load_data(i_episode)
      t1 = time.process_time()
      if not (self.debug and i_episode == 0):
        agent_logs = self.agent.update()
      else: agent_logs = {}
      t2 = time.process_time()

      timing_logs = {
        "learn_time" : t2 - t1,
        "collection_time" : t1 - t0,
      }

      # perform logging and saving
      test_logs = {}
      if i_episode % self.save_freq == 0: self.save(i_episode)
      if i_episode % self.test_freq == 0:
        if i_episode != 0 or self.debug: 
          test_logs = self.test(i_episode)

      logging_dict = {
        "Agent" : agent_logs,
        "Performance" : timing_logs,
        "Test" : test_logs,
      }
      self.log(i_episode, log_dict=logging_dict)

    # save the final state of the model
    self.save(self.num_episodes)
    
  def log(self, i_episode, log_dict={}, pad=35):
    """
    Log training output given the logger stored at self.logger, which is expected to
    be a ProjectLogger, from utils.logger.ProjectLogger
    """

    collection_time = log_dict["Performance"]["collection_time"]
    learn_time = log_dict["Performance"]["learn_time"]
    num_envs = self.agent.storage.num_envs

    self.tot_timesteps += self.num_env_steps_per_update * num_envs
    self.tot_time += collection_time + learn_time
    iteration_time = collection_time + learn_time

    log_dict["Performance"]["fps"] = int(self.num_env_steps_per_update * num_envs / \
              (collection_time + learn_time))
    
    # recursively log each of the leaves in the dictionary
    self.logger.log_dict(log_dict)

    curr_it = i_episode - self.start_learning_iteration
    eta = self.tot_time / (curr_it + 1) * (self.num_episodes - curr_it)
    mins = eta // 60
    secs = eta % 60
    log_string = (f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                  f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                  f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                  f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")

    self.logger.log_step(print_string=log_string)

  def save(self, i_episode):
    """
    Save the state of the agent (saves a new file each time, numbered by episode)
    """

    if not self.enable_saving: 
      pylogger.info("Trainer.save(): self.enable_saving = False, nothing saved")
      return

    self.modelsaver.save(self.agent.name, pyobj=self.agent.get_save_state(),
                         force_suffix=i_episode)

  def test(self, i_episode):
    """
    Evaluate anomaly detection on the released real-world trajectories, and return
    a dictionary of metrics.
    """

    if not self.use_evaulator:
      pylogger.info(f"use_evaluator=False, no test run at episode={i_episode}.")
      return {}

    pylogger.info(f"Running a discrimination test at episode={i_episode}")

    # wrap the agent into discriminators
    dummy_latents = not self.agent.model.add_privileged_info
    disc_latent = Agent_Latent_Discriminator(agent=self.agent, dummy_latents=dummy_latents)
    disc_pred = Agent_Prediction_Discriminator(agent=self.agent)
    disc_double = Agent_Double_Discriminator(latent_discrim=disc_latent, 
                                             pred_discrim=disc_pred)

    test_logs = {}
    for case, trajectory in self.eval_set.items():
      eval_dict = anomaly.evaluate(disc_double, trajectory, name=f"Model i_episode={i_episode}",
                                   traj_label=case)
      for key in eval_dict:
        test_logs[f"{case}_{key}_f1"] = eval_dict[key]["f1"]
        test_logs[f"{case}_{key}_average_precision"] = eval_dict[key]["average_precision"]
        test_logs[f"{case}_{key}_auroc"] = eval_dict[key]["auroc"]

    return test_logs
