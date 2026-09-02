#!/usr/bin/env python3
"""Convert a pinned TinyLlama Safetensors checkpoint to the E2 FP16 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


MANIFEST_VERSION = 1
DEFAULT_REVISION = "88e648e026e00c9252e31312b7ba8eb2fdd9b7c3"
ALIGNMENT = 256


def expected_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    kv_size = int(config["num_key_value_heads"]) * (
        hidden // int(config["num_attention_heads"])
    )
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (int(config["vocab_size"]), hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (int(config["vocab_size"]), hidden),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (hidden,),
                prefix + "post_attention_layernorm.weight": (hidden,),
                prefix + "self_attn.q_proj.weight": (hidden, hidden),
                prefix + "self_attn.k_proj.weight": (kv_size, hidden),
                prefix + "self_attn.v_proj.weight": (kv_size, hidden),
                prefix + "self_attn.o_proj.weight": (hidden, hidden),
                prefix + "mlp.gate_proj.weight": (intermediate, hidden),
                prefix + "mlp.up_proj.weight": (intermediate, hidden),
                prefix + "mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return shapes


def validate_config(config: dict) -> None:
    required = {
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    mismatches = [
        f"{key}: expected {value!r}, got {config.get(key)!r}"
        for key, value in required.items()
        if config.get(key) != value
    ]
    if float(config.get("rms_norm_eps", 0.0)) != 1.0e-5:
        mismatches.append("rms_norm_eps must be 1e-5")
    if float(config.get("rope_theta", 0.0)) != 10000.0:
        mismatches.append("rope_theta must be 10000")
    if mismatches:
        raise ValueError("checkpoint is not the fixed TinyLlama baseline:\n"
                         + "\n".join(mismatches))


def aligned(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def shape_text(shape: Iterable[int]) -> str:
    return ",".join(str(value) for value in shape)


def convert(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.resolve()
    config_path = model_dir / "config.json"
    source_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not source_path.is_file():
        raise FileNotFoundError(
            "model directory must contain config.json and model.safetensors"
        )
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    shapes = expected_shapes(config)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "tinyllama-1.1b-chat-fp16.weights"
    manifest_path = output_dir / "tinyllama-1.1b-chat-fp16.manifest"
    if not args.force and (data_path.exists() or manifest_path.exists()):
        raise FileExistsError("output exists; pass --force to replace it")
    data_tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    records: list[tuple[str, int, int, tuple[int, ...], str]] = []
    try:
        with safe_open(source_path, framework="pt", device="cpu") as source:
            source_keys = set(source.keys())
            if source_keys != set(shapes):
                missing = sorted(set(shapes) - source_keys)
                unexpected = sorted(source_keys - set(shapes))
                raise ValueError(
                    f"checkpoint tensor set mismatch; missing={missing}, "
                    f"unexpected={unexpected}"
                )
            with data_tmp.open("wb") as output:
                cursor = 0
                for index, (name, wanted_shape) in enumerate(shapes.items(), 1):
                    target = aligned(cursor)
                    if target != cursor:
                        output.write(bytes(target - cursor))
                    tensor = source.get_tensor(name)
                    if tuple(tensor.shape) != wanted_shape:
                        raise ValueError(
                            f"{name}: expected {wanted_shape}, got "
                            f"{tuple(tensor.shape)}"
                        )
                    tensor = tensor.to(dtype=torch.float16).contiguous()
                    byte_view = memoryview(tensor.numpy()).cast("B")
                    output.write(byte_view)
                    digest = hashlib.sha256(byte_view).hexdigest()
                    records.append(
                        (name, target, len(byte_view), wanted_shape, digest)
                    )
                    cursor = target + len(byte_view)
                    print(f"[{index:03d}/{len(shapes):03d}] {name}")
                output.flush()
                os.fsync(output.fileno())
        os.replace(data_tmp, data_path)

        head_dimension = int(config["hidden_size"]) // int(
            config["num_attention_heads"]
        )
        lines = [
            f"version={MANIFEST_VERSION}",
            "checkpoint=TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            f"checkpoint_revision={args.checkpoint_revision}",
            f"tokenizer_revision={args.tokenizer_revision}",
            f"config_sha256={hashlib.sha256(config_bytes).hexdigest()}",
            "dtype=fp16",
            f"data_file={data_path.name}",
            f"data_bytes={data_path.stat().st_size}",
            f"hidden_size={config['hidden_size']}",
            f"intermediate_size={config['intermediate_size']}",
            f"layer_count={config['num_hidden_layers']}",
            f"attention_head_count={config['num_attention_heads']}",
            f"kv_head_count={config['num_key_value_heads']}",
            f"head_dimension={head_dimension}",
            f"vocabulary_size={config['vocab_size']}",
            f"max_position_embeddings={config['max_position_embeddings']}",
            f"bos_token_id={config['bos_token_id']}",
            f"eos_token_id={config['eos_token_id']}",
            f"rms_norm_epsilon={config['rms_norm_eps']}",
            f"rope_theta={config['rope_theta']}",
            f"tied_word_embeddings={int(config['tie_word_embeddings'])}",
            f"tensor_count={len(records)}",
        ]
        lines.extend(
            f"tensor={name}|{offset}|{size}|{shape_text(shape)}|{digest}"
            for name, offset, size, shape, digest in records
        )
        manifest_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
    finally:
        data_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)

    print(f"manifest: {manifest_path}")
    print(f"weights:  {data_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", default=DEFAULT_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_REVISION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
