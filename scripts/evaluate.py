"""
Evaluate anomaly detection of a trained model on the released crane dataset, reporting
average precision, AUROC and F1 score for the latent, predictor and combined (double)
anomaly signals.

  python scripts/evaluate.py checkpoint=outputs/<date>/<time> id=150
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl") # headless rendering, set before importing mujoco

import logging
import hydra
from omegaconf import DictConfig

from aware.agents.double_discriminator import Agent_Double_Discriminator
from aware.evaluation import anomaly
from aware.utils.logger import LoggingLevel, create_table

COLUMNS = {
  "AP" : "average_precision",
  "AUROC" : "auroc",
  "F1 score" : "f1",
  "SR" : "sr",
  "Hz" : "approx_frequency",
}

@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):

  checkpoint = os.path.abspath(cfg.checkpoint)
  name = cfg.name or os.path.basename(os.path.normpath(checkpoint))
  if cfg.id is not None:
    name += f" (id={cfg.id})"

  rows = []
  for dataset in cfg.datasets:

    # load the model afresh for each dataset, since RSSM AWARE predictions are sampled
    with LoggingLevel(logging.WARNING): # silence many log statements
      double_discrim = Agent_Double_Discriminator(timestamp=checkpoint, id=cfg.id, device=cfg.device,
                                                  auto_infer_load_args=True)

    print(f"Running evaluation on {name} now (dataset={dataset})")
    results = anomaly.evaluate(double_discrim, anomaly.load_dataset(dataset), name=name,
                               traj_label=dataset)
    for signal, metrics in results.items():
      rows.append([name, signal, dataset] + [metrics[key] for key in COLUMNS.values()])

  # summarise the results in a table
  headings = ["Name", "Type", "Dataset"] + list(COLUMNS)
  header_str, row_str = create_table(
    widths=[max(len(headings[i]), *(len(row[i]) for row in rows)) for i in range(3)]
            + [9 for _ in COLUMNS],
    types=[str, str, str] + [float for _ in COLUMNS],
    float_fmt=".3f",
  )
  table_str = header_str.format(*headings) + "".join(row_str.format(*row) for row in rows)
  print(f"\n--- Evaluation results ---\n{table_str}")

  os.makedirs(os.path.dirname(os.path.abspath(cfg.output)), exist_ok=True)
  with open(cfg.output, "w") as f:
    f.write(table_str)
  print(f"Results saved to {cfg.output}")

if __name__ == "__main__":
  main()
