#!/usr/bin/env python3
"""Compare E3 runtime greedy output with an independent Transformers model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM


def parse_manifest(path: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    metadata: dict[str, str] = {}
    tensors: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key != "tensor":
            metadata[key] = value
            continue
        name, offset, byte_size, shape, _sha256 = value.split("|", 4)
        tensors.append(
            {
                "name": name,
                "offset": int(offset),
                "byte_size": int(byte_size),
                "shape": tuple(int(item) for item in shape.split(",")),
            }
        )
    return metadata, tensors


def load_model(manifest_path: Path, weights_path: Path) -> LlamaForCausalLM:
    metadata, tensors = parse_manifest(manifest_path)
    config = LlamaConfig(
        hidden_size=int(metadata["hidden_size"]),
        intermediate_size=int(metadata["intermediate_size"]),
        num_hidden_layers=int(metadata["layer_count"]),
        num_attention_heads=int(metadata["attention_head_count"]),
        num_key_value_heads=int(metadata["kv_head_count"]),
        head_dim=int(metadata["head_dimension"]),
        vocab_size=int(metadata["vocabulary_size"]),
        max_position_embeddings=int(metadata["max_position_embeddings"]),
        bos_token_id=int(metadata["bos_token_id"]),
        eos_token_id=int(metadata["eos_token_id"]),
        rms_norm_eps=float(metadata["rms_norm_epsilon"]),
        rope_theta=float(metadata["rope_theta"]),
        tie_word_embeddings=metadata["tied_word_embeddings"] == "1",
        attention_bias=False,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    element_count = int(metadata["data_bytes"]) // torch.float16.itemsize
    flat_weights = torch.from_file(
        str(weights_path), shared=False, size=element_count, dtype=torch.float16
    )
    state_dict = {
        str(tensor["name"]): flat_weights.narrow(
            0,
            int(tensor["offset"]) // torch.float16.itemsize,
            int(tensor["byte_size"]) // torch.float16.itemsize,
        ).view(tensor["shape"])
        for tensor in tensors
    }
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device("cuda"):
            model = LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    incompatible = model.load_state_dict(state_dict, strict=True, assign=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"state_dict mismatch: {incompatible}")
    model = model.eval()
    model.config._attn_implementation = "eager"
    return model


def main(args: argparse.Namespace) -> None:
    runtime = json.loads(args.runtime_json.read_text(encoding="utf-8"))
    model = load_model(args.manifest, args.weights)
    input_ids = torch.tensor(
        [runtime["input_tokens"]], dtype=torch.long, device="cuda"
    )
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=int(runtime["max_new_tokens"]),
            do_sample=False,
            eos_token_id=int(runtime["eos_token_id"]),
            pad_token_id=int(runtime["eos_token_id"]),
            use_cache=True,
        )
    reference_tokens = generated[0, input_ids.shape[1] :].cpu().tolist()
    runtime_tokens = [int(token) for token in runtime["output_tokens"]]
    report = {
        "checkpoint": runtime["checkpoint"],
        "checkpoint_revision": runtime["checkpoint_revision"],
        "input_length": len(runtime["input_tokens"]),
        "max_new_tokens": int(runtime["max_new_tokens"]),
        "eos_token_id": int(runtime["eos_token_id"]),
        "runtime_tokens": runtime_tokens,
        "reference_tokens": reference_tokens,
        "tokens_equal": runtime_tokens == reference_tokens,
        "terminal_reason": runtime["terminal_reason"],
        "usage": runtime["usage"],
        "metrics_ns": runtime["metrics_ns"],
        "repetitions": int(runtime["repetitions"]),
        "outputs_consistent": bool(runtime["outputs_consistent"]),
        "resources_reclaimed": bool(runtime["resources_reclaimed"]),
        "gpu_free_bytes_delta": int(runtime["gpu_free_bytes_delta"]),
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "reference_dtype": "float16",
            "attention_implementation": "eager",
            "reference_prefill": "batched",
            "runtime_prefill": "serial_token",
        },
    }
    report["passed"] = (
        report["tokens_equal"]
        and report["outputs_consistent"]
        and report["resources_reclaimed"]
        and report["gpu_free_bytes_delta"] == 0
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
