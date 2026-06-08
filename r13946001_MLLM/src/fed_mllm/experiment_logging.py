from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from fed_mllm.model import create_model, set_parameters


CSV_FIELDS = [
    "method",
    "round_or_event",
    "server_version",
    "arrival_server_version",
    "simulated_time",
    "client_id",
    "client_start_version",
    "staleness",
    "base_alpha",
    "effective_alpha",
    "staleness_weight",
    "learning_rate",
    "train_loss",
    "test_loss",
    "test_acc",
    "num_examples",
    "delay",
    "buffer_size",
    "applied_updates",
    "agreement",
    "mean_agreement",
    "buffer_alpha",
    "delta_norm",
    "dropped_update",
    "server_momentum_agreement",
    "fairness_weight",
    "pending_pool_size",
    "quorum_size",
    "quorum_met",
    "selected_updates",
    "invalid_answer_rate",
]


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ExperimentLogger:
    """Small CSV logger for reproducible course-project experiments."""

    def __init__(
        self,
        method: str,
        *,
        result_dir: str | Path = "results",
        dataset_name: str = "mmlu",
        run_id: str | None = None,
    ) -> None:
        self.method = method
        self.dataset_name = dataset_name
        self.run_id = run_id or make_run_id()
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.result_dir / f"{method}_{dataset_name}_{self.run_id}.csv"
        self.summary_path = (
            self.result_dir / f"{method}_{dataset_name}_{self.run_id}_summary.json"
        )
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()

    def log(self, **row: Any) -> None:
        payload = {field: row.get(field, "") for field in CSV_FIELDS}
        self._writer.writerow(payload)
        self._file.flush()

    def write_summary(self, summary: dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

    def close(self) -> None:
        self._file.close()


def save_checkpoint(
    *,
    parameters: list,
    method: str,
    dataset_name: str = "mmlu",
    best_test_acc: float,
    best_round_or_event: int,
    args_config: dict[str, Any],
    checkpoint_dir: str | Path = "checkpoints",
    run_id: str,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{method}_{dataset_name}_{run_id}_best.pt"

    num_classes = int(args_config.get("num_classes", 4))
    input_mode = str(args_config.get("input_mode", "text"))
    model_name = str(args_config.get("model", "tiny_text"))
    # For LoRA/QLoRA models, storing full base weights would create very large
    # checkpoints. The federated server only aggregates adapter parameters, so
    # save those directly for reproducibility and push hygiene.
    if model_name in {"qwen_text_0_5b_lora", "qwen2_5_vl_3b_qlora"}:
        torch.save(
            {
                "adapter_parameters": parameters,
                "method": method,
                "best_test_acc": best_test_acc,
                "best_round_or_event": best_round_or_event,
                "args": args_config,
            },
            path,
        )
        return path

    model = create_model(
        num_classes=num_classes,
        input_mode=input_mode,
        model_name=model_name,
    )
    set_parameters(model, parameters)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": method,
            "best_test_acc": best_test_acc,
            "best_round_or_event": best_round_or_event,
            "args": args_config,
        },
        path,
    )
    return path
