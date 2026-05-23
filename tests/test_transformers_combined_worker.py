from __future__ import annotations

from scripts.run_transformers_math_eq_combined_worker import (
    apply_chat_template_safe,
    build_combined_result,
    resolve_prompt_pad_id,
)


class TokenizerWithoutThinking:
    pad_token_id = None
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("unsupported keyword")
        assert tokenize is False
        assert add_generation_prompt is True
        return "prompt<think>\n"

    def __call__(self, text, add_special_tokens=False):
        assert text == " "
        assert add_special_tokens is False
        return {"input_ids": [123]}


class TokenizerWithThinking:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False
        return "prompt"


def test_apply_chat_template_safe_falls_back_and_strips_forced_think():
    prompt = apply_chat_template_safe(
        TokenizerWithoutThinking(),
        [{"role": "user", "content": "hello"}],
        think_seed_mode="closed_direct",
        think_seed_text="direct",
    )
    assert prompt == "prompt<think>direct</think>\n"


def test_apply_chat_template_safe_uses_enable_thinking_when_supported():
    prompt = apply_chat_template_safe(
        TokenizerWithThinking(),
        [{"role": "user", "content": "hello"}],
        think_seed_mode="off",
    )
    assert prompt == "prompt"


def test_resolve_prompt_pad_id_prefers_space_token():
    assert resolve_prompt_pad_id(TokenizerWithoutThinking(), None) == 123
    assert resolve_prompt_pad_id(TokenizerWithoutThinking(), 456) == 456


def test_build_combined_result_balances_math_and_eq_scores():
    combined = build_combined_result(
        config_key=(0, 1, 1, 2),
        layer_indices=[0, 1, 1, 2],
        config_spec="layers:0,1,1,2",
        math_result={"score": 0.25, "valid_final_answer_count": 3, "valid_final_answer_rate": 0.75},
        eq_result={"score": 0.75},
        elapsed=12.5,
        math_batch_size=16,
        eq_batch_size=8,
        math_retries=1,
        eq_retries=0,
        metadata={"loader": "AutoModelForCausalLM"},
    )
    assert combined["combined_score"] == 0.5
    assert combined["mode"] == "transformers_separate_math_eq"
    assert combined["math_batch_size"] == 16
    assert combined["eq_batch_size"] == 8
    assert combined["metadata"]["loader"] == "AutoModelForCausalLM"
