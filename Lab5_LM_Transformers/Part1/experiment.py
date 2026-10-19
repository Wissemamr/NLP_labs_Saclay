"""Training, evaluation, and report artifacts for the Part 1 notebook."""
import csv
import json
import logging
import math
import platform
import time
import uuid
from importlib.metadata import version
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch.nn import functional as F


def timestamp():
    return datetime.now(ZoneInfo("Australia/Brisbane")).isoformat()


def perplexity(loss):
    return math.exp(loss) if loss < 709 else float("inf")


def prepare_batch(tokenizer, texts, max_length, device):
    encoded = tokenizer(texts, return_tensors="pt", padding=True,
                        max_length=max_length, truncation=True)
    labels = encoded.input_ids[:, 1:].clone()
    labels[encoded.attention_mask[:, 1:] == 0] = -100
    return (encoded.input_ids[:, :-1].to(device), labels.to(device),
            encoded.attention_mask[:, :-1].to(device))


@torch.inference_mode()
def evaluate(model, loader, tokenizer, max_length, device):
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    try:
        for batch in loader:
            x, labels, mask = prepare_batch(tokenizer, batch["text"], max_length, device)
            _, logits = model(x, attention_mask=mask)
            total_nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                        labels.reshape(-1), ignore_index=-100,
                                        reduction="sum").item()
            total_tokens += (labels != -100).sum().item()
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("Validation contains no next-token targets")
    loss = total_nll / total_tokens
    if not math.isfinite(loss):
        raise RuntimeError("Nonfinite validation loss")
    return {"loss": loss, "perplexity": perplexity(loss), "target_tokens": total_tokens}


def train_run(model, optimizer, training_dl, validation_dl, tokenizer, *,
              model_config, runs_dir, max_length, max_steps, num_epochs,
              eval_every, seed, tokenizer_id):
    """Persist each metric immediately; save a final checkpoint on successful completion.

    max_steps is a global update limit across all epochs, not a per-epoch limit.
    Call with a fresh model/optimizer for an independent experiment.
    """
    if min(max_steps, num_epochs, eval_every) < 1:
        raise ValueError("max_steps, num_epochs, and eval_every must be positive")
    if not len(training_dl) or not len(validation_dl):
        raise ValueError("Training and validation loaders must be nonempty")
    run_id = datetime.now(ZoneInfo("Australia/Brisbane")).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(runs_dir) / (run_id + "_" + uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    metadata = dict(run_id=run_dir.name, started_at=timestamp(), timezone="Australia/Brisbane",
                    status="running", seed=seed, device=str(device), python=platform.python_version(),
                    torch=str(torch.__version__), model_config=model_config,
                    library_versions={name: version(name) for name in
                                      ("transformers", "datasets", "matplotlib")},
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    dataset="roneneldan/TinyStories", subset_selection="first N per split; training shuffled",
                    tokenizer_id=tokenizer_id, selected_training_samples=len(training_dl.dataset),
                    validation_samples=len(validation_dl.dataset), batch_size=training_dl.batch_size,
                    max_length=max_length, configured_epochs=num_epochs, max_steps=max_steps,
                    planned_steps=min(max_steps, num_epochs * len(training_dl)), eval_every=eval_every,
                    optimizer=type(optimizer).__name__,
                    optimizer_settings={k: v for k, v in optimizer.param_groups[0].items() if k != "params"},
                    initialization="fresh random weights", steps_completed=0,
                    samples_processed=0, target_tokens_processed=0, epochs_completed=0.0)
    logger = logging.getLogger(f"part1.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    fields = ["split", "step", "epoch", "samples_processed", "target_tokens_processed",
              "loss", "perplexity", "target_tokens", "learning_rate", "gradient_norm", "elapsed_seconds"]
    history = []
    start = time.perf_counter()

    def save_metadata():
        temporary = run_dir / "metadata.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(run_dir / "metadata.json")

    def record(split, step, result, writer, handle, gradient_norm=""):
        row = dict(split=split, step=step, epoch=metadata["epochs_completed"],
                   samples_processed=metadata["samples_processed"],
                   target_tokens_processed=metadata["target_tokens_processed"],
                   loss=result["loss"], perplexity=result["perplexity"],
                   target_tokens=result["target_tokens"], learning_rate=optimizer.param_groups[0]["lr"],
                   gradient_norm=gradient_norm, elapsed_seconds=time.perf_counter() - start)
        writer.writerow(row)
        handle.flush()
        history.append(row)
        logger.info("%s | step %d/%d | epoch %.3f | loss %.4f | perplexity %.2f",
                    split, step, metadata["planned_steps"], row["epoch"], row["loss"], row["perplexity"])

    try:
        save_metadata()
        tokenizer.save_pretrained(run_dir / "tokenizer")
        # Preserve the exact model definitions/configuration used by this run.
        source_dir = Path(__file__).parent
        sources = {}
        for name in ("implement_transformer.ipynb", "rope_embedding.py", "experiment.py"):
            sources[name] = (source_dir / name).read_text(encoding="utf-8")
        (run_dir / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
        logger.info("%s | started %s | artifacts: %s", metadata["run_id"], metadata["started_at"], run_dir)
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            result = evaluate(model, validation_dl, tokenizer, max_length, device)
            record("validation", 0, result, writer, handle)
            metadata["initial_validation"] = result
            step = 0
            model.train()
            for epoch in range(num_epochs):
                for batch in training_dl:
                    x, labels, mask = prepare_batch(tokenizer, batch["text"], max_length, device)
                    target_tokens = (labels != -100).sum().item()
                    if not target_tokens:
                        raise ValueError("Training batch contains no next-token targets")
                    optimizer.zero_grad(set_to_none=True)
                    _, logits = model(x, attention_mask=mask)
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                           ignore_index=-100)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                                   error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    metadata["steps_completed"] = step
                    metadata["samples_processed"] += len(batch["text"])
                    metadata["unique_training_samples_seen"] = min(
                        metadata["samples_processed"], len(training_dl.dataset))
                    metadata["target_tokens_processed"] += target_tokens
                    metadata["epochs_completed"] = metadata["samples_processed"] / len(training_dl.dataset)
                    record("train", step, {"loss": loss.item(), "perplexity": perplexity(loss.item()),
                                          "target_tokens": target_tokens}, writer, handle,
                           gradient_norm=float(gradient_norm))
                    if step % eval_every == 0 or step == metadata["planned_steps"]:
                        result = evaluate(model, validation_dl, tokenizer, max_length, device)
                        record("validation", step, result, writer, handle)
                        metadata["final_validation"] = result
                    save_metadata()
                    if step >= max_steps:
                        break
                if step >= max_steps:
                    break
        checkpoint = dict(model_config=model_config, model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(), step=step,
                          tokenizer_id=tokenizer_id, run_id=metadata["run_id"],
                          loss_history=[r["loss"] for r in history if r["split"] == "train"],
                          torch_rng_state=torch.get_rng_state())
        torch.save(checkpoint, run_dir / "model.pt.tmp")
        (run_dir / "model.pt.tmp").replace(run_dir / "model.pt")
        metadata["status"] = "completed"
        metadata["checkpoint"] = "model.pt"
        logger.info("Completed | checkpoint: %s", run_dir / "model.pt")
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = str(error)
        logger.exception("Run did not complete")
        raise
    finally:
        metadata["finished_at"] = timestamp()
        metadata["elapsed_seconds"] = time.perf_counter() - start
        save_metadata()
        for handler in (file_handler, stream_handler):
            handler.close()
            logger.removeHandler(handler)
    return run_dir, history, metadata


def plot_run(run_dir):
    """Plot measured curves; persist standalone figures for a report."""
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    with (run_dir / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for split, label, color in (("train", "Training batches (before update)", "tab:blue"),
                                ("validation", "Held-out validation (after update)", "tab:orange")):
        selected = [row for row in rows if row["split"] == split]
        for ax, metric in zip(axes, ("loss", "perplexity")):
            ax.plot([int(r["step"]) for r in selected], [float(r[metric]) for r in selected],
                    label=label, color=color, linewidth=1.2,
                    marker="o" if split == "validation" else None, markersize=4,
                    alpha=1 if split == "validation" else 0.6)
    axes[0].set_ylabel("Next-token cross-entropy loss")
    axes[1].set_ylabel("Perplexity (log scale)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Training run {run_dir.name}", fontsize=10)
    fig.savefig(run_dir / "learning_curves.png", dpi=200)
    fig.savefig(run_dir / "learning_curves.pdf")
    return fig
