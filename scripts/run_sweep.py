from __future__ import annotations

import csv
import itertools
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
SWEEP_CONFIG_DIR = CONFIG_DIR / "sweeps"
SWEEP_REPORT_DIR = RESULTS_DIR / "sweeps"
TRAIN_LOG_PATH = RESULTS_DIR / "train_log_v2.csv"


def load_base_config() -> dict:
    base_config_path = CONFIG_DIR / "default.yaml"
    with base_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_command(cmd: list[str], log_path: Path, label: str) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(f"[{label}] {line}")
            sys.stdout.flush()
            f.write(line)
            output_lines.append(line)
        return process.wait(), "".join(output_lines)


def find_new_run_dir(before: set[str], after: set[str]) -> Path:
    new_dirs = sorted(after - before)
    if not new_dirs:
        raise RuntimeError("No new checkpoint directory was created by training.")
    if len(new_dirs) > 1:
        raise RuntimeError(f"Expected one new checkpoint directory, found: {new_dirs}")
    return CHECKPOINT_DIR / new_dirs[0]


def append_report_line(report_path: Path, text: str = "") -> None:
    with report_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def parse_eval_metrics(output_text: str) -> dict[str, float] | None:
    patterns = {
        "recall@1": r"\|\s*Recall@1\s*\|\s*([0-9.]+)\s*\|",
        "recall@5": r"\|\s*Recall@5\s*\|\s*([0-9.]+)\s*\|",
        "recall@10": r"\|\s*Recall@10\s*\|\s*([0-9.]+)\s*\|",
        "mAP": r"\|\s*mAP\s*\|\s*([0-9.]+)\s*\|",
    }
    metrics: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if match is None:
            return None
        metrics[key] = float(match.group(1))
    return metrics


def append_sweep_summary(summary_path: Path, row: dict[str, str | int | float]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_train_rows_for_run(run_id: str) -> list[dict[str, str]]:
    if not TRAIN_LOG_PATH.exists():
        raise FileNotFoundError(f"Missing train log: {TRAIN_LOG_PATH}")

    with TRAIN_LOG_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("run_id") == run_id]

    if not rows:
        raise RuntimeError(f"No train log rows found for run_id={run_id}")
    return rows


def select_best_train_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: float(row["mAP"]))


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = f"proxyNCA_embed_lr_{timestamp}"
    report_path = SWEEP_REPORT_DIR / f"{sweep_name}.md"
    summary_path = SWEEP_REPORT_DIR / f"{sweep_name}.csv"
    SWEEP_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_base_config()
    loss_method = "proxyNCA"
    learning_rates = [5e-5, 1e-4, 2e-4]
    embed_dims = [64, 128, 256]
    weight_decay = 0.0

    append_report_line(report_path, f"# Sweep Report: {sweep_name}")
    append_report_line(report_path, "")
    append_report_line(report_path, "## Settings")
    append_report_line(report_path, f"- embed_dims: `{embed_dims}`")
    append_report_line(report_path, f"- learning_rates: `{learning_rates}`")
    append_report_line(report_path, f"- weight_decay: `{weight_decay}`")
    append_report_line(report_path, f"- loss_method: `{loss_method}`")
    append_report_line(report_path, "- selection metric during training: best epoch by train-time mAP")
    append_report_line(report_path, "- external evaluation: best checkpoint only")
    append_report_line(report_path, f"- summary csv: `{summary_path}`")
    append_report_line(report_path, "")

    combos = list(itertools.product(embed_dims, learning_rates))
    total_runs = len(combos)

    for run_idx, (embed_dim, learning_rate) in enumerate(combos, start=1):
        combo_name = f"loss_{loss_method}_ed_{embed_dim}_lr_{learning_rate}".replace(".", "p")
        combo_config = yaml.safe_load(yaml.safe_dump(base_config))
        combo_config["model"]["embedding_dim"] = embed_dim
        combo_config["train"]["learning_rate"] = float(learning_rate)
        combo_config["train"]["weight_decay"] = float(weight_decay)
        combo_config["train"]["loss_method"] = loss_method
        combo_config["runtime"]["device"] = "auto"

        config_path = SWEEP_CONFIG_DIR / f"{sweep_name}_{combo_name}.yaml"
        write_config(combo_config, config_path)

        append_report_line(report_path, f"## Run: `{combo_name}`")
        append_report_line(report_path, f"- config: `{config_path}`")
        append_report_line(report_path, "")
        print(f"\n=== [{run_idx}/{total_runs}] Training {combo_name} ===")
        print(f"Config: {config_path}")

        before_dirs = {p.name for p in CHECKPOINT_DIR.iterdir() if p.is_dir()} if CHECKPOINT_DIR.exists() else set()

        train_log_path = SWEEP_REPORT_DIR / f"{sweep_name}_{combo_name}_train.log"
        train_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "train.py"),
            "--config",
            str(config_path),
        ]
        train_returncode, _ = run_command(train_cmd, train_log_path, f"train:{combo_name}")

        append_report_line(report_path, "### Train")
        append_report_line(report_path, f"- command: ``{' '.join(train_cmd)}``")
        append_report_line(report_path, f"- log: `{train_log_path}`")
        append_report_line(report_path, f"- return code: `{train_returncode}`")

        if train_returncode != 0:
            append_report_line(report_path, "")
            append_report_line(report_path, "```text")
            append_report_line(report_path, f"Training failed. Check log: {train_log_path}")
            append_report_line(report_path, "```")
            append_report_line(report_path, "")
            continue

        after_dirs = {p.name for p in CHECKPOINT_DIR.iterdir() if p.is_dir()}
        run_dir = find_new_run_dir(before_dirs, after_dirs)
        train_rows = load_train_rows_for_run(run_dir.name)
        best_train_row = select_best_train_row(train_rows)
        best_checkpoint = run_dir / "best.pth"
        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Best checkpoint missing: {best_checkpoint}")
        print(
            f"Completed training for {combo_name}. "
            f"Best checkpoint is {best_checkpoint.name} (epoch {best_train_row['epoch']}, "
            f"train mAP {best_train_row['mAP']})."
        )

        append_report_line(report_path, f"- run_dir: `{run_dir}`")
        append_report_line(report_path, f"- best checkpoint by train-time mAP: `{best_checkpoint}`")
        append_report_line(report_path, f"- best train mAP: `{best_train_row['mAP']}`")
        append_report_line(report_path, "")
        append_report_line(report_path, "### Evaluate")
        eval_log_path = SWEEP_REPORT_DIR / f"{sweep_name}_{combo_name}_{best_checkpoint.stem}.log"
        eval_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "evaluate.py"),
            "--checkpoint",
            str(best_checkpoint),
        ]
        print(f"  -> Evaluating best checkpoint: {best_checkpoint.name}")
        eval_returncode, eval_output = run_command(
            eval_cmd,
            eval_log_path,
            f"eval:{combo_name}:{best_checkpoint.stem}",
        )
        metrics = parse_eval_metrics(eval_output)

        append_report_line(report_path, f"#### `{best_checkpoint.name}`")
        append_report_line(report_path, f"- command: ``{' '.join(eval_cmd)}``")
        append_report_line(report_path, f"- log: `{eval_log_path}`")
        append_report_line(report_path, "```text")
        with eval_log_path.open("r", encoding="utf-8") as f:
            append_report_line(report_path, f.read().rstrip())
        append_report_line(report_path, "```")
        append_report_line(report_path, "")
        append_report_line(report_path, f"- return code: `{eval_returncode}`")
        append_report_line(report_path, "")

        if eval_returncode == 0 and metrics is not None:
            train_cfg = combo_config["train"]
            append_sweep_summary(
                summary_path,
                {
                    "sweep_name": sweep_name,
                    "run_id": run_dir.name,
                    "config_path": str(config_path),
                    "combo_name": combo_name,
                    "loss_method": loss_method,
                    "embedding_dim": embed_dim,
                    "learning_rate": float(train_cfg["learning_rate"]),
                    "backbone_learning_rate": (
                        float(train_cfg["backbone_learning_rate"])
                        if train_cfg.get("backbone_learning_rate") is not None
                        else float(train_cfg["learning_rate"])
                    ),
                    "head_learning_rate": (
                        float(train_cfg["head_learning_rate"])
                        if train_cfg.get("head_learning_rate") is not None
                        else float(train_cfg["learning_rate"])
                    ),
                    "proxy_learning_rate": (
                        float(train_cfg["proxy_learning_rate"])
                        if train_cfg.get("proxy_learning_rate") is not None
                        else float(train_cfg["learning_rate"])
                    ),
                    "scheduler": str(train_cfg.get("scheduler", "none")),
                    "scheduler_step_size": train_cfg.get("scheduler_step_size"),
                    "scheduler_gamma": float(train_cfg.get("scheduler_gamma", 0.1)),
                    "scheduler_min_lr": float(train_cfg.get("scheduler_min_lr", 1e-6)),
                    "batch_size": int(train_cfg["batch_size"]),
                    "num_workers": int(combo_config["data"]["num_workers"]),
                    "epochs": int(train_cfg["epochs"]),
                    "sampler_p": int(train_cfg["sampler_p"]),
                    "sampler_k": int(train_cfg["sampler_k"]),
                    "weight_decay": float(train_cfg["weight_decay"]),
                    "best_checkpoint": str(best_checkpoint),
                    "best_epoch": int(best_train_row["epoch"]),
                    "best_train_loss": float(best_train_row["train_loss"]),
                    "best_train_mAP": float(best_train_row["mAP"]),
                    "recall@1": metrics["recall@1"],
                    "recall@5": metrics["recall@5"],
                    "recall@10": metrics["recall@10"],
                    "eval_mAP": metrics["mAP"],
                },
            )


if __name__ == "__main__":
    main()
