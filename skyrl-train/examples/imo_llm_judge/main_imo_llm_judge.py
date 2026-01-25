"""
Main entrypoint for the IMO AnswerBench LLM-as-a-judge example.
"""

import ray
import hydra
from omegaconf import DictConfig

from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register the imo_llm_judge environment inside the entrypoint task (no need to modify skyrl-gym).
    register(
        id="imo_llm_judge",
        entry_point="examples.imo_llm_judge.imo_judge_env:IMOAnswerBenchLLMJudgeEnv",
    )

    # Debug: show the effective batch size after Hydra overrides (common source of confusion).
    print(f"[imo_llm_judge] cfg.trainer.train_batch_size={cfg.trainer.train_batch_size}", flush=True)
    print(f"[imo_llm_judge] train_data={cfg.data.train_data}", flush=True)

    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()

