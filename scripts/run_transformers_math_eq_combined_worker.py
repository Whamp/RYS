#!/usr/bin/env python
"""Combined Hugging Face Transformers Math+EQ scanner.

This is the BF16/safetensors counterpart to the ExLlama combined worker. It
loads a Transformers model once, iterates a canonical RYS queue, evaluates Math
and EQ for each layer configuration, and writes Math, EQ, and combined result
pickles incrementally.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _strip_forced_think(prompt: str) -> str:
    if prompt.endswith("<think>\n"):
        return prompt[: -len("<think>\n")]
    if prompt.endswith("<think>"):
        return prompt[: -len("<think>")]
    return prompt


def _append_think_seed(prompt: str, think_seed_mode: str, think_seed_text: str) -> str:
    if think_seed_mode == "off":
        return prompt
    if think_seed_mode == "closed_direct":
        return f"{prompt}<think>{think_seed_text}</think>\n"
    raise ValueError(f"Unknown think seed mode: {think_seed_mode}")


def apply_chat_template_safe(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    think_seed_mode: str = "off",
    think_seed_text: str = "I can answer this now, and will do so succinctly.",
) -> str:
    """Apply a tokenizer chat template, falling back if enable_thinking is unsupported."""
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return _append_think_seed(_strip_forced_think(prompt), think_seed_mode, think_seed_text)


def pretokenize_messages(
    *,
    dataset: dict[str, Any],
    tokenizer: Any,
    device: Any,
    message_builder,
    payload_builder,
    use_no_think_prefix: bool,
    think_seed_mode: str,
    think_seed_text: str,
) -> dict[str, dict[str, Any]]:
    """Pretokenize a dataset through message/payload builder callbacks."""
    tokenized: dict[str, dict[str, Any]] = {}
    for qid, sample in dataset.items():
        messages = message_builder(sample, use_no_think_prefix=use_no_think_prefix)
        prompt = apply_chat_template_safe(
            tokenizer,
            messages,
            think_seed_mode=think_seed_mode,
            think_seed_text=think_seed_text,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        payload = payload_builder(sample)
        tokenized[qid] = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
            **payload,
        }
    return tokenized


def resolve_prompt_pad_id(tokenizer: Any, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    try:
        space_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
        if space_ids:
            return int(space_ids[-1])
    except Exception:
        pass
    return int(tokenizer.pad_token_id or tokenizer.eos_token_id)


def build_combined_result(
    *,
    config_key: tuple[int, ...],
    layer_indices: list[int],
    config_spec: str,
    math_result: dict[str, Any],
    eq_result: dict[str, Any],
    elapsed: float,
    math_batch_size: int,
    eq_batch_size: int,
    math_retries: int,
    eq_retries: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    math_score = float(math_result.get("score", 0.0))
    eq_score = float(eq_result.get("score", 0.0))
    return {
        "config_key": config_key,
        "config_layers": list(layer_indices),
        "config_spec": config_spec,
        "elapsed": elapsed,
        "mode": "transformers_separate_math_eq",
        "math_score": math_score,
        "eq_score": eq_score,
        "combined_score": 0.5 * (math_score + eq_score),
        "math_batch_size": int(math_batch_size),
        "eq_batch_size": int(eq_batch_size),
        "math_retries": int(math_retries),
        "eq_retries": int(eq_retries),
        "math_valid_final_answer_count": math_result.get("valid_final_answer_count"),
        "math_valid_final_answer_rate": math_result.get("valid_final_answer_rate"),
        "metadata": dict(metadata),
    }


def _save_pickle_result(path: Path, config_key: tuple[int, ...], value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("wb") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                pickle.dump({}, f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    with path.open("r+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            data = pickle.load(f)
            data[config_key] = value
            f.seek(0)
            f.truncate()
            pickle.dump(data, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def cuda_memory_snapshot() -> dict[str, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
            "max_allocated": int(torch.cuda.max_memory_allocated()),
            "max_reserved": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return None


def assert_layer_metadata(run_model: Any, layer_indices: list[int], source_num_layers: int) -> None:
    """Fail early if duplicated layer metadata is inconsistent for Qwen3.5 masks/cache."""
    if layer_indices == list(range(source_num_layers)):
        return
    context = getattr(run_model, "_apply_layer_config", None)
    if context is None:
        return
    with context():
        base_model = run_model.base_model
        owner = run_model._layers_owner
        attr = run_model._layers_attr
        expanded_layers = getattr(owner, attr)
        target_num_layers = len(layer_indices)
        if len(expanded_layers) != target_num_layers:
            raise RuntimeError(f"Expanded layer count mismatch: {len(expanded_layers)} != {target_num_layers}")

        text_cfg = getattr(base_model.config, "text_config", None)
        layer_types = getattr(text_cfg, "layer_types", None) if text_cfg is not None else None
        if layer_types is None:
            layer_types = getattr(base_model.config, "layer_types", None)
        if layer_types is not None:
            if len(layer_types) != target_num_layers:
                raise RuntimeError(f"layer_types length mismatch: {len(layer_types)} != {target_num_layers}")
            for pos, layer in enumerate(expanded_layers):
                layer_type = getattr(layer, "layer_type", None)
                if layer_type is not None and layer_type != layer_types[pos]:
                    raise RuntimeError(
                        f"layer_types[{pos}]={layer_types[pos]!r} does not match layer.layer_type={layer_type!r}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined Transformers BF16/safetensors Math+EQ scanner.")
    parser.add_argument("--queue-file", required=True)
    parser.add_argument("--combined-results-file", required=True)
    parser.add_argument("--math-results-file", required=True)
    parser.add_argument("--eq-results-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--math-dataset-path", default="datasets/math_16.json")
    parser.add_argument("--eq-dataset-path", default="datasets/eq_16.json")
    parser.add_argument("--math-batch-size", type=int, default=16)
    parser.add_argument("--eq-batch-size", type=int, default=8)
    parser.add_argument("--math-max-new", type=int, default=64)
    parser.add_argument("--eq-max-new", type=int, default=64)
    parser.add_argument("--padding-mode", choices=["masked", "inprompt_space"], default="masked")
    parser.add_argument("--prompt-pad-id", type=int, default=None)
    parser.add_argument("--adaptive-batch-retry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-retries-per-phase", type=int, default=8)
    parser.add_argument("--attention-impl", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--max-memory-json", default=None)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-causal-loader", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-no-think-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--think-seed-mode", choices=["off", "closed_direct"], default="closed_direct")
    parser.add_argument("--think-seed-text", default="I can answer this now, and will do so succinctly.")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-samples", type=int, default=4)
    parser.add_argument("--preflight-max-new", type=int, default=64)
    parser.add_argument("--preflight-worst-span", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.padding_mode == "inprompt_space":
        raise SystemExit("Qwen3.5/Qwen3.6 linear-attention scans must use --padding-mode masked.")
    if args.force_causal_loader:
        os.environ["LEVELGEN_TEXT_LOADER"] = "causal"

    import torch
    import transformers

    from src.core.layer_config import expand_single_block, is_baseline_layers, layer_spec_string, parse_queue_entry_layers
    from src.core.layer_duplicator import build_model_with_layers
    from src.core.layer_duplicator_moe import build_model_with_layers_moe
    from src.workers.batch_control import adaptive_batch_execute
    from src.workers.eq_worker import generate_eq_messages, run_eq_preflight, run_eq_test
    from src.workers.math_worker import (
        generate_messages as generate_math_messages,
        run_math_preflight,
        run_math_test_batched_moe,
    )
    from src.workers.model_utils import is_moe_model, load_model_and_tokenizer, parse_device_map_arg, parse_max_memory_json
    from src.workers.shared_queue import SharedWorkQueue, format_eta

    resolved_device_map = parse_device_map_arg(args.device_map)
    resolved_max_memory = parse_max_memory_json(args.max_memory_json)
    worker_id = args.worker_id or f"TF-COMB-{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"

    print("=" * 80)
    print(f"Transformers Combined Worker [{worker_id}]")
    print("=" * 80)
    print(f"Model:                 {args.model_path}")
    print(f"Queue file:            {args.queue_file}")
    print(f"Combined results file: {args.combined_results_file}")
    print(f"Math results file:     {args.math_results_file}")
    print(f"EQ results file:       {args.eq_results_file}")
    print(f"Math dataset:          {args.math_dataset_path}")
    print(f"EQ dataset:            {args.eq_dataset_path}")
    print(f"Batch sizes:           math={args.math_batch_size}, eq={args.eq_batch_size}")
    print(f"Max new tokens:        math={args.math_max_new}, eq={args.eq_max_new}")
    print(f"Padding mode:          {args.padding_mode}")
    print(f"Attention impl:        {args.attention_impl}")
    print(f"Force causal loader:   {args.force_causal_loader}")
    print(f"Transformers version:  {transformers.__version__}")

    print("\nLoading datasets...")
    with open(args.math_dataset_path, "r") as f:
        math_dataset = json.load(f)
    with open(args.eq_dataset_path, "r") as f:
        eq_dataset = json.load(f)
    print(f"Loaded {len(math_dataset)} math + {len(eq_dataset)} EQ prompts")

    print("\nLoading model once...")
    attn_impl = args.attention_impl if args.attention_impl != "eager" else None
    tokenizer, model, load_meta = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map=resolved_device_map,
        attn_implementation=attn_impl,
        max_memory=resolved_max_memory,
        cpu_offload=args.cpu_offload,
        offload_folder=args.offload_folder,
    )
    if args.force_causal_loader and load_meta["loader"] != "AutoModelForCausalLM":
        raise RuntimeError(f"Expected AutoModelForCausalLM loader, got {load_meta['loader']}")
    print(f"Loader:                {load_meta['loader']}")
    print(f"Architectures:         {load_meta['architectures']}")
    print(f"Text stack:            {load_meta['text_stack']}")
    print(f"HF device map:         {load_meta.get('hf_device_map')}")
    print(f"Memory after load:     {cuda_memory_snapshot()}")

    num_layers = int(load_meta["num_layers"])
    model_is_moe = is_moe_model(model)
    build_duplicated_model = build_model_with_layers_moe if model_is_moe else build_model_with_layers
    print(f"Model type:            {'MoE' if model_is_moe else 'Dense'}")
    print(f"Text layers:           {num_layers}")

    print("\nPretokenizing datasets...")

    def math_messages(sample: dict[str, Any], *, use_no_think_prefix: bool) -> list[dict[str, str]]:
        return generate_math_messages(sample["question"], use_no_think_prefix=use_no_think_prefix)

    def math_payload(sample: dict[str, Any]) -> dict[str, Any]:
        return {"answer": sample["answer"]}

    def eq_messages(sample: dict[str, Any], *, use_no_think_prefix: bool) -> list[dict[str, str]]:
        return generate_eq_messages(sample["prompt"], use_no_think_prefix=use_no_think_prefix)

    def eq_payload(sample: dict[str, Any]) -> dict[str, Any]:
        return {"reference": sample.get("reference_answer", sample.get("reference_answer_fullscale", {}))}

    tokenized_math = pretokenize_messages(
        dataset=math_dataset,
        tokenizer=tokenizer,
        device=model.device,
        message_builder=math_messages,
        payload_builder=math_payload,
        use_no_think_prefix=args.use_no_think_prefix,
        think_seed_mode=args.think_seed_mode,
        think_seed_text=args.think_seed_text,
    )
    tokenized_eq = pretokenize_messages(
        dataset=eq_dataset,
        tokenizer=tokenizer,
        device=model.device,
        message_builder=eq_messages,
        payload_builder=eq_payload,
        use_no_think_prefix=args.use_no_think_prefix,
        think_seed_mode=args.think_seed_mode,
        think_seed_text=args.think_seed_text,
    )
    prompt_pad_id = resolve_prompt_pad_id(tokenizer, args.prompt_pad_id)

    def run_math_with_retry(
        run_model: Any,
        *,
        max_new: int | None = None,
        batch_size: int | None = None,
        dataset_override: dict[str, dict[str, Any]] | None = None,
    ):
        execution = adaptive_batch_execute(
            lambda batch: run_math_test_batched_moe(
                run_model,
                dataset_override or tokenized_math,
                tokenizer,
                batch_size=batch,
                max_new_tokens=max_new or args.math_max_new,
                save_responses=True,
                padding_mode=args.padding_mode,
                prompt_pad_id=prompt_pad_id,
            ),
            initial_batch_size=batch_size or args.math_batch_size,
            min_batch_size=args.min_batch_size,
            max_retries=args.max_retries_per_phase,
            enabled=args.adaptive_batch_retry,
            phase_name="math",
            on_retry=lambda msg: print(f"[{worker_id}] {msg}"),
        )
        return execution.result, execution.batch_size, execution.retries

    def run_eq_with_retry(
        run_model: Any,
        *,
        max_new: int | None = None,
        batch_size: int | None = None,
        dataset_override: dict[str, dict[str, Any]] | None = None,
    ):
        execution = adaptive_batch_execute(
            lambda batch: run_eq_test(
                run_model,
                dataset_override or tokenized_eq,
                tokenizer,
                batch_size=batch,
                max_new_tokens=max_new or args.eq_max_new,
                save_responses=True,
                padding_mode=args.padding_mode,
                prompt_pad_id=prompt_pad_id,
            ),
            initial_batch_size=batch_size or args.eq_batch_size,
            min_batch_size=args.min_batch_size,
            max_retries=args.max_retries_per_phase,
            enabled=args.adaptive_batch_retry,
            phase_name="eq",
            on_retry=lambda msg: print(f"[{worker_id}] {msg}"),
        )
        return execution.result, execution.batch_size, execution.retries

    if not args.skip_preflight:
        print("\nRunning preflight...")
        math_preflight = run_math_preflight(
            model,
            tokenized_math,
            tokenizer,
            samples=min(args.preflight_samples, len(tokenized_math)),
            batch_size=args.math_batch_size,
            max_new_tokens=args.preflight_max_new,
            padding_mode=args.padding_mode,
            prompt_pad_id=prompt_pad_id,
            min_extract_rate=0.5,
        )
        eq_preflight = run_eq_preflight(
            model,
            tokenized_eq,
            tokenizer,
            samples=min(args.preflight_samples, len(tokenized_eq)),
            batch_size=args.eq_batch_size,
            max_new_tokens=args.preflight_max_new,
            padding_mode=args.padding_mode,
            prompt_pad_id=prompt_pad_id,
            min_nonzero_conf_rate=0.5,
        )
        print(f"Preflight baseline math={math_preflight['score']:.4f}, eq={eq_preflight['score']:.4f}")

        if args.preflight_worst_span and num_layers > 1:
            worst_layers = expand_single_block(num_layers, (0, num_layers))
            dup_model = build_duplicated_model(model, worst_layers)
            assert_layer_metadata(dup_model, worst_layers, num_layers)
            sample_count = max(1, min(args.preflight_samples, len(tokenized_math), len(tokenized_eq)))
            math_subset = dict(list(tokenized_math.items())[:sample_count])
            eq_subset = dict(list(tokenized_eq.items())[:sample_count])
            print(f"Preflight worst-span layers={len(worst_layers)} samples={sample_count} at batch=1")
            _math_result, _math_batch, _math_retries = run_math_with_retry(
                dup_model,
                max_new=args.preflight_max_new,
                batch_size=1,
                dataset_override=math_subset,
            )
            _eq_result, _eq_batch, _eq_retries = run_eq_with_retry(
                dup_model,
                max_new=args.preflight_max_new,
                batch_size=1,
                dataset_override=eq_subset,
            )
            print(
                "Preflight worst-span "
                f"math={_math_result['score']:.4f}, eq={_eq_result['score']:.4f}, "
                f"memory={cuda_memory_snapshot()}"
            )
            del dup_model
            torch.cuda.empty_cache()

    queue = SharedWorkQueue(args.queue_file, args.combined_results_file)
    metadata = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "loader": load_meta["loader"],
        "architectures": load_meta["architectures"],
        "model_type": load_meta["model_type"],
        "text_stack": load_meta["text_stack"],
        "num_layers": num_layers,
        "attention_impl": args.attention_impl,
        "padding_mode": args.padding_mode,
        "think_seed_mode": args.think_seed_mode,
    }

    print("\nStarting queue loop...")
    configs_processed = 0
    start_time = time.time()
    while True:
        entry = queue.get_next_config()
        if entry is None:
            print("Queue empty. Exiting.")
            break
        try:
            parsed_entry = parse_queue_entry_layers(num_layers, entry)
        except Exception as exc:
            print(f"[{worker_id}] Invalid queue entry {entry!r}: {exc}")
            continue

        layer_indices = parsed_entry["layers"]
        config_key = parsed_entry["layer_key"]
        config_spec = parsed_entry["spec"]
        remaining, completed = queue.get_queue_status()
        print(
            f"\n[{worker_id}] Running {config_spec} ({layer_spec_string(layer_indices)}) "
            f"remaining={remaining}, completed={completed}"
        )

        t0 = time.time()
        if is_baseline_layers(layer_indices, num_layers):
            run_model = model
        else:
            run_model = build_duplicated_model(model, layer_indices)
            assert_layer_metadata(run_model, layer_indices, num_layers)

        try:
            math_result, math_batch, math_retries = run_math_with_retry(run_model)
            eq_result, eq_batch, eq_retries = run_eq_with_retry(run_model)
        finally:
            if run_model is not model:
                del run_model
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        combined = build_combined_result(
            config_key=config_key,
            layer_indices=layer_indices,
            config_spec=config_spec,
            math_result=math_result,
            eq_result=eq_result,
            elapsed=elapsed,
            math_batch_size=math_batch,
            eq_batch_size=eq_batch,
            math_retries=math_retries,
            eq_retries=eq_retries,
            metadata={**metadata, "cuda_memory": cuda_memory_snapshot()},
        )
        queue.save_result(config_key, combined)
        _save_pickle_result(Path(args.math_results_file), config_key, math_result)
        _save_pickle_result(Path(args.eq_results_file), config_key, eq_result)

        configs_processed += 1
        total_elapsed = time.time() - start_time
        rate = configs_processed / total_elapsed if total_elapsed > 0 else 0.0
        eta = format_eta(remaining / rate) if remaining > 0 and rate > 0 else "N/A"
        print(
            f"[{worker_id}] math={combined['math_score']:.4f} eq={combined['eq_score']:.4f} "
            f"combined={combined['combined_score']:.4f} elapsed={elapsed:.1f}s "
            f"batch=({math_batch},{eq_batch}) retries=({math_retries},{eq_retries}) "
            f"rate={rate:.3f}/s ETA={eta}"
        )

    print(f"Processed {configs_processed} configs in {format_eta(time.time() - start_time)}")


if __name__ == "__main__":
    main()
