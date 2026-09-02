#!/usr/bin/env python3
"""Compare E2 runtime tensors and logits with a Transformers FP16 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1.0e-5)
    return {
        "max_error_index": int(difference.reshape(-1).argmax().item()),
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "max_relative_error": (difference / denominator).max().item(),
    }


def main(args: argparse.Namespace) -> None:
    runtime = json.loads(args.runtime_json.read_text(encoding="utf-8"))
    tokens = torch.tensor(
        [runtime["input_tokens"]], dtype=torch.long, device="cuda"
    )
    selected_layers = [int(value) for value in runtime["selected_layers"]]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_index in selected_layers:
        def capture(_module, _inputs, output, index=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[:, -1, :].detach().float().cpu()

        hooks.append(model.model.layers[layer_index].register_forward_hook(capture))
    with torch.inference_mode():
        embedding = model.model.embed_tokens(tokens)[:, -1, :].float().cpu()
        output = model(
            input_ids=tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    for hook in hooks:
        hook.remove()

    runtime_embedding = torch.tensor(runtime["embedding"])[None, :]
    runtime_layers = [
        torch.tensor(values)[None, :]
        for values in runtime["layer_hidden_states"]
    ]
    runtime_final = torch.tensor(runtime["final_hidden_state"])[None, :]
    runtime_logits = torch.tensor(runtime["logits"])
    reference_final = output.hidden_states[-1][:, -1, :].float().cpu()
    reference_logits = output.logits[0, -1, :].float().cpu()

    runtime_greedy = [int(value) for value in runtime["greedy_tokens"]]
    reference_greedy = output.logits[0].argmax(dim=-1).cpu().tolist()
    top_k = 10
    runtime_top = runtime_logits.topk(top_k).indices.tolist()
    reference_top = reference_logits.topk(top_k).indices.tolist()

    report = {
        "checkpoint": runtime["checkpoint"],
        "checkpoint_revision": runtime["checkpoint_revision"],
        "config_sha256": runtime["config_sha256"],
        "input_tokens": runtime["input_tokens"],
        "selected_layers": selected_layers,
        "embedding": metrics(runtime_embedding, embedding),
        "layers": {
            str(layer): metrics(actual, captured[layer])
            for layer, actual in zip(selected_layers, runtime_layers)
        },
        "final_hidden_state": metrics(runtime_final, reference_final),
        "logits": metrics(runtime_logits, reference_logits),
        "runtime_greedy_tokens": runtime_greedy,
        "reference_greedy_tokens": reference_greedy,
        "runtime_top10": runtime_top,
        "reference_top10": reference_top,
        "top10_overlap": len(set(runtime_top) & set(reference_top)),
        "resource_reclaimed": (
            runtime["released_request_count"] == 0
            and runtime["released_committed_token_count"] == 0
        ),
        "runtime_memory": {
            "device_weight_bytes": runtime["device_weight_bytes"],
            "device_workspace_bytes": runtime["device_workspace_bytes"],
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "reference_dtype": "float16",
            "attention_implementation": "eager",
        },
        "thresholds": {
            "hidden_max_absolute_error": args.hidden_max_error,
            "logits_max_absolute_error": args.logits_max_error,
            "logits_mean_absolute_error": args.logits_mean_error,
            "required_top10_overlap": args.top10_overlap,
            "all_greedy_tokens_equal": True,
        },
    }
    hidden_results = [report["embedding"], *report["layers"].values(),
                      report["final_hidden_state"]]
    report["passed"] = (
        all(item["max_absolute_error"] <= args.hidden_max_error
            for item in hidden_results)
        and report["logits"]["max_absolute_error"] <= args.logits_max_error
        and report["logits"]["mean_absolute_error"] <= args.logits_mean_error
        and report["top10_overlap"] >= args.top10_overlap
        and runtime_greedy == reference_greedy
        and report["resource_reclaimed"]
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-max-error", type=float, default=0.05)
    parser.add_argument("--logits-max-error", type=float, default=0.05)
    parser.add_argument("--logits-mean-error", type=float, default=0.01)
    parser.add_argument("--top10-overlap", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
