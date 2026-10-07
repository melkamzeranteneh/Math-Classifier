"""Built-in single-device fine-tuning adapter for Laya.

The adapter targets the public ``laya.train.finetune`` API.  It deliberately
does not reimplement Laya's model, loss, calibration, or checkpoint format.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def train(
    records: Sequence[Mapping[str, Any]],
    output_path: str,
    *,
    device: str = "cuda:0",
    model_dir: str = "convaiinnovations/laya",
    checkpoint_dir: str | None = None,
    epochs: int = 4,
    micro_batch: int = 8,
    grad_accum: int = 8,
    loss: str = "rlcd",
    seed: int = 42,
    freeze_encoder: bool = False,
    shuffle_options: bool = False,
    eval_data: str | None = None,
    max_len: int | None = None,
    head_max_len: int | None = None,
) -> dict[str, Any]:
    """Fine-tune Laya and return its training report.

    ``records`` must be the records produced by :mod:`laya_retrain`.  The
    saved directory is loadable by ``laya.load`` and is separate from the
    prepared JSONL file.
    """
    if not records:
        raise ValueError("cannot fine-tune Laya with an empty record set")
    effective_device = _normalise_cuda_device(device)
    try:
        from laya.train import TrainConfig, finetune
    except ImportError as exc:
        raise RuntimeError(
            "This Laya installation does not expose laya.train.finetune. "
            "Install a Laya release with the fine-tuning API (laya>=0.3.28)."
        ) from exc

    prepared_path = Path(output_path)
    train_path = prepared_path.with_name(prepared_path.stem + ".finetune.jsonl")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    destination = Path(checkpoint_dir) if checkpoint_dir else prepared_path.with_name(
        prepared_path.stem + "-checkpoint"
    )
    config_kwargs: dict[str, Any] = {
        "epochs": epochs,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "loss": loss,
        "seed": seed,
        "freeze_encoder": freeze_encoder,
    }
    if shuffle_options:
        config_kwargs["shuffle_options"] = ("choice",)
    if eval_data:
        config_kwargs["eval_data"] = str(Path(eval_data))
    if max_len is not None:
        config_kwargs["max_len"] = max_len
    if head_max_len is not None:
        config_kwargs["head_max_len"] = head_max_len

    config = TrainConfig(**config_kwargs)
    report = finetune(
        data=str(train_path),
        model_dir=model_dir,
        output_dir=str(destination),
        config=config,
        device=effective_device,
    )
    if not isinstance(report, dict):
        raise RuntimeError(
            f"Laya fine-tuning returned {type(report).__name__}, expected a report dictionary"
        )
    report_path = destination / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return {
        "checkpoint_dir": str(destination),
        "training_data": str(train_path),
        "report": report,
    }


def _normalise_cuda_device(device: str) -> str:
    """Avoid Laya 0.3.x multi-index placement bugs.

    Laya's trainer reliably uses the logical ``cuda`` device.  When the caller
    requests a physical GPU and CUDA visibility is not already configured,
    remap that GPU to logical index zero before importing Laya/PyTorch.
    """
    if not device.startswith("cuda:"):
        return device
    index = device[5:]
    if not index.isdigit():
        raise ValueError(f"invalid CUDA device {device!r}")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = index
        return "cuda"
    return "cuda"
