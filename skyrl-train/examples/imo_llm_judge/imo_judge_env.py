from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.aime import utils as aime_utils
from typing import Any, Dict
from omegaconf import DictConfig
from openai import OpenAI
import os
import re


JUDGE_PROMPT = """You are a strict math evaluation assistant.

Given the following extracted answer and ground truth answer, determine if they are mathematically equivalent.

You must respond with exactly one token on a single line:
- CORRECT
- INCORRECT
"""


def _extract_last_boxed(answer_text: str) -> str | None:
    boxed = aime_utils.last_boxed_only_string(answer_text)
    if boxed is None:
        return None
    try:
        return aime_utils.remove_boxed(boxed)
    except Exception:
        return None


class IMOAnswerBenchLLMJudgeEnv(BaseTextEnv):
    """IMO AnswerBench environment: require boxed answer, then use an LLM judge for equivalence."""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"

        self.ground_truth = str(extras["reward_spec"]["ground_truth"]).strip()

        # Config
        self.model = getattr(env_config, "model", "gpt-4o-mini")
        self.base_url = getattr(env_config, "base_url", None)
        self.require_boxed = bool(getattr(env_config, "require_boxed", True))

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("`OPENAI_API_KEY` must be set for IMO LLM judge env")
        self.llm_judge_client = OpenAI(base_url=self.base_url, api_key=openai_api_key)

    def _judge_equivalence(self, pred: str, gt: str) -> tuple[bool, str]:
        prompt = (
            JUDGE_PROMPT
            + f"\nAnswer: {pred}\nGround truth: {gt}\n"
        )
        response = self.llm_judge_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = (response.choices[0].message.content or "").strip()

        # Robust parsing: accept exact or line-containing token.
        if re.search(r"\bINCORRECT\b", reply, flags=re.IGNORECASE):
            return False, reply
        if re.search(r"\bCORRECT\b", reply, flags=re.IGNORECASE):
            return True, reply
        return False, reply

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True

        extracted = _extract_last_boxed(action) if self.require_boxed else action.strip()
        if self.require_boxed and extracted is None:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=done,
                metadata={"format_ok": False, "extracted_pred": None},
            )

        ok, judge_reply = self._judge_equivalence(extracted, self.ground_truth)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=1.0 if ok else 0.0,
            done=done,
            metadata={
                "format_ok": True,
                "extracted_pred": extracted,
                "judge_reply": judge_reply,
            },
        )

