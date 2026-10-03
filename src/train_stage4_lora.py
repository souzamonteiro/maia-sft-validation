from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)

from src.common_stage4 import *
from src.dataset_stage4 import (
    ResponseOnlyDataset,
    Collator,
)


# ============================================================
# MAIA SFT VALIDATION
# STAGE 4 - FULL ASSISTANT CONTROL
#
# GPT-2 MEDIUM RESPONSE-ONLY LoRA SFT
#
# Features:
#
#   - visible tqdm progress inside every epoch
#   - instantaneous loss
#   - running mean loss
#   - current learning rate
#   - global optimizer update
#   - ETA
#   - periodic resumable LoRA checkpoints
#   - automatic exact resume
#   - model state
#   - optimizer state
#   - scheduler state
#   - Python RNG state
#   - PyTorch RNG state
#   - MPS RNG state when available
#   - validation after every epoch
#   - best-model preservation
#   - persistent metrics
#   - frozen Stage 4 V4 corpus verification
#
# IMPORTANT:
#
# Checkpoints are resumable LoRA training checkpoints.
# Best/final directories contain LoRA adapters plus tokenizer files.
# Merge/export is intentionally a separate deployment step.
# ============================================================


# ============================================================
# FROZEN STAGE 4 V4 CORPUS
# ============================================================


FROZEN_HASHES = {
    "train.jsonl":
        "ee80a1efc74807d354761d9728cd498e9f543c84a5e568e4f3821d2fa48565f9",

    "validation.jsonl":
        "5d8ea877e14105e06068f2d7de5ae55d265b4069af2c1b0762b162df1be1347a",

    "test.jsonl":
        "03b7289a5841f79e1c95188f22615dca6789846aeddecc1b3c90f9713c918834",
}


# ============================================================
# TRAINING CHECKPOINT CONFIGURATION
# ============================================================


CHECKPOINT_EVERY_UPDATES = 250

CHECKPOINT_ROOT = OUT / "stage4-lora-checkpoints"

BEST_DIR = OUT / "gpt2-medium-stage4-lora-best"

FINAL_DIR = OUT / "gpt2-medium-stage4-lora-final"

METRICS_FILE = OUT / "stage4-lora-training-metrics.json"

STATE_FILE_NAME = "training_state.pt"


# ============================================================
# LoRA CONFIGURATION - FROZEN FOR THIS EXPERIMENT
# ============================================================


LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]
LORA_BIAS = "none"


def lora_experiment_config():
    return {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": list(LORA_TARGET_MODULES),
        "bias": LORA_BIAS,
        "task_type": "CAUSAL_LM",
    }


# ============================================================
# UTILITIES
# ============================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def verify_frozen_corpus():
    print()
    print("=" * 80)
    print("VERIFYING FROZEN STAGE 4 V4 CORPUS")
    print("=" * 80)

    for filename, expected in FROZEN_HASHES.items():
        path = DATA / filename

        if not path.exists():
            raise RuntimeError(
                f"Frozen corpus file not found: {path}"
            )

        actual = sha256_file(path)

        print()
        print(filename)
        print("  expected:", expected)
        print("  actual:  ", actual)

        if actual != expected:
            raise RuntimeError(
                f"Frozen corpus hash mismatch: {filename}"
            )

    print()
    print("Frozen corpus verification: PASS")


def save_json_atomic(path: Path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def remove_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)


def empty_device_cache(dev):
    """Release cached accelerator memory when supported."""
    if dev.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def require_finite_loss(loss, *, phase, batch_index=None):
    """Abort immediately if a non-finite loss is observed."""
    if not torch.isfinite(loss.detach()).item():
        location = f" at batch {batch_index}" if batch_index is not None else ""
        raise RuntimeError(
            f"Non-finite loss detected during {phase}{location}: "
            f"{float(loss.detach().cpu())}"
        )


# ============================================================
# RNG STATE
# ============================================================


def capture_rng_state():
    state = {
        "python":
            random.getstate(),

        "torch":
            torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    # MPS RNG functions depend on PyTorch version.
    if (
        hasattr(torch, "mps")
        and hasattr(
            torch.mps,
            "get_rng_state",
        )
    ):
        try:
            state["mps"] = (
                torch.mps.get_rng_state()
            )
        except Exception:
            pass

    return state


def restore_rng_state(state):
    if not state:
        return

    if "python" in state:
        random.setstate(
            state["python"]
        )

    if "torch" in state:
        torch.set_rng_state(
            state["torch"]
        )

    if (
        "cuda" in state
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            state["cuda"]
        )

    if (
        "mps" in state
        and hasattr(torch, "mps")
        and hasattr(
            torch.mps,
            "set_rng_state",
        )
    ):
        try:
            torch.mps.set_rng_state(
                state["mps"]
            )
        except Exception:
            pass


# ============================================================
# DATA LOADERS
# ============================================================


def make_train_loader(
    dataset,
    collator,
    epoch: int,
):
    """
    Build a deterministic shuffle for each epoch.

    Recreating the loader with the same epoch number recreates
    exactly the same sample order. This allows safe mid-epoch
    resume by skipping batches already consumed.
    """

    generator = torch.Generator()

    generator.manual_seed(
        SEED + epoch
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )


def make_validation_loader(
    dataset,
    collator,
):
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )


# ============================================================
# VALIDATION
# ============================================================


def evaluate(
    model,
    loader,
    dev,
    description="Validation",
):
    model.eval()

    loss_sum = 0.0
    batches = 0

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch in progress:
            batch = {
                key: value.to(dev)
                for key, value
                in batch.items()
            }

            loss = model(
                **batch
            ).loss

            require_finite_loss(
                loss,
                phase=description,
            )

            value = float(
                loss.detach().cpu()
            )

            loss_sum += value
            batches += 1

            progress.set_postfix(
                loss=f"{value:.4f}",
                avg=(
                    f"{loss_sum / batches:.4f}"
                ),
                refresh=False,
            )

    return (
        loss_sum
        / max(1, batches)
    )


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================


def checkpoint_update_number(
    path: Path,
):
    try:
        return int(
            path.name.split("-")[-1]
        )
    except Exception:
        return -1


def find_latest_checkpoint():
    if not CHECKPOINT_ROOT.exists():
        return None

    candidates = []

    for path in CHECKPOINT_ROOT.glob(
        "checkpoint-*"
    ):
        if not path.is_dir():
            continue

        state_file = (
            path / STATE_FILE_NAME
        )

        if not state_file.exists():
            continue

        candidates.append(path)

    if not candidates:
        return None

    candidates.sort(
        key=checkpoint_update_number
    )

    return candidates[-1]


# ============================================================
# CHECKPOINT SAVE
# ============================================================


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    epoch,
    batch_in_epoch,
    global_update,
    total_updates,
    best_validation_loss,
    initial_validation_loss,
    history,
    epoch_loss_sum,
    epoch_loss_batches,
):
    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = (
        CHECKPOINT_ROOT
        / f"checkpoint-{global_update:06d}"
    )

    temporary_path = (
        CHECKPOINT_ROOT
        / f".checkpoint-{global_update:06d}.tmp"
    )

    remove_directory(
        temporary_path
    )

    temporary_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Model + tokenizer
    # --------------------------------------------------------

    model.save_pretrained(
        temporary_path,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        temporary_path
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    state = {
        "epoch":
            epoch,

        "batch_in_epoch":
            batch_in_epoch,

        "global_update":
            global_update,

        "total_updates":
            total_updates,

        "best_validation_loss":
            best_validation_loss,

        "initial_validation_loss":
            initial_validation_loss,

        "history":
            history,

        "epoch_loss_sum":
            epoch_loss_sum,

        "epoch_loss_batches":
            epoch_loss_batches,

        "optimizer":
            optimizer.state_dict(),

        "scheduler":
            scheduler.state_dict(),

        "rng":
            capture_rng_state(),

        "seed":
            SEED,

        "config": {
            "model_id":
                CFG["model_id"],

            "max_length":
                CFG["max_length"],

            "learning_rate":
                CFG["learning_rate"],

            "weight_decay":
                CFG["weight_decay"],

            "gradient_accumulation_steps":
                CFG[
                    "gradient_accumulation_steps"
                ],

            "epochs":
                CFG["epochs"],

            "warmup_ratio":
                CFG["warmup_ratio"],

            "max_grad_norm":
                CFG["max_grad_norm"],
        },

        "frozen_hashes":
            FROZEN_HASHES,

        "lora_config":
            lora_experiment_config(),
    }

    torch.save(
        state,
        temporary_path
        / STATE_FILE_NAME,
    )

    # --------------------------------------------------------
    # Atomic directory publication
    # --------------------------------------------------------

    remove_directory(
        final_path
    )

    temporary_path.replace(
        final_path
    )

    print()
    print(
        f"Checkpoint saved: "
        f"{final_path}"
    )

    return final_path


# ============================================================
# CHECKPOINT LOAD
# ============================================================


def load_checkpoint_state(
    checkpoint: Path,
    dev,
):
    state_file = (
        checkpoint
        / STATE_FILE_NAME
    )

    if not state_file.exists():
        raise RuntimeError(
            f"Missing training state: "
            f"{state_file}"
        )

    state = torch.load(
        state_file,
        map_location="cpu",
        weights_only=False,
    )

    if (
        state.get("frozen_hashes")
        != FROZEN_HASHES
    ):
        raise RuntimeError(
            "Checkpoint belongs to a different "
            "Stage 4 corpus."
        )

    if (
        state.get("lora_config")
        != lora_experiment_config()
    ):
        raise RuntimeError(
            "Checkpoint belongs to a different "
            "Stage 4 LoRA configuration."
        )

    return state


# ============================================================
# METRICS
# ============================================================


def write_metrics(
    initial_validation_loss,
    best_validation_loss,
    global_update,
    total_updates,
    history,
    status,
):
    metrics = {
        "status":
            status,

        "initial_validation_loss":
            initial_validation_loss,

        "best_validation_loss":
            best_validation_loss,

        "optimizer_updates":
            global_update,

        "total_optimizer_updates":
            total_updates,

        "history":
            history,

        "frozen_corpus_sha256":
            FROZEN_HASHES,

        "lora_config":
            lora_experiment_config(),
    }

    save_json_atomic(
        METRICS_FILE,
        metrics,
    )


# ============================================================
# MAIN
# ============================================================


def main():
    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed_all()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Frozen corpus
    # --------------------------------------------------------

    verify_frozen_corpus()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    dev = device()

    print()
    print("=" * 80)
    print("STAGE 4 LoRA TRAINING")
    print("=" * 80)

    print()
    print("Device:", dev)
    print("Model: ", CFG["model_id"])

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            CFG["model_id"]
        )
    )

    tokenizer.pad_token = (
        tokenizer.pad_token
        or tokenizer.eos_token
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    train_rows = read_jsonl(
        DATA / "train.jsonl"
    )

    validation_rows = read_jsonl(
        DATA / "validation.jsonl"
    )

    train_dataset = (
        ResponseOnlyDataset(
            train_rows,
            tokenizer,
            CFG["max_length"],
        )
    )

    validation_dataset = (
        ResponseOnlyDataset(
            validation_rows,
            tokenizer,
            CFG["max_length"],
        )
    )

    collator = Collator(
        tokenizer
    )

    validation_loader = (
        make_validation_loader(
            validation_dataset,
            collator,
        )
    )

    print(
        "Training examples:  ",
        f"{len(train_dataset):,}",
    )

    print(
        "Validation examples:",
        f"{len(validation_dataset):,}",
    )

    # --------------------------------------------------------
    # Training dimensions
    # --------------------------------------------------------

    accumulation_steps = (
        CFG[
            "gradient_accumulation_steps"
        ]
    )

    batches_per_epoch = (
        len(train_dataset)
    )

    updates_per_epoch = math.ceil(
        batches_per_epoch
        / accumulation_steps
    )

    total_updates = (
        updates_per_epoch
        * CFG["epochs"]
    )

    warmup_updates = max(
        1,
        int(
            total_updates
            * CFG["warmup_ratio"]
        ),
    )

    print()
    print(
        "Epochs:                ",
        CFG["epochs"],
    )

    print(
        "Batches/epoch:         ",
        f"{batches_per_epoch:,}",
    )

    print(
        "Gradient accumulation: ",
        accumulation_steps,
    )

    print(
        "Updates/epoch:         ",
        f"{updates_per_epoch:,}",
    )

    print(
        "Total updates:         ",
        f"{total_updates:,}",
    )

    print(
        "Warmup updates:        ",
        f"{warmup_updates:,}",
    )

    print(
        "Checkpoint interval:   ",
        f"{CHECKPOINT_EVERY_UPDATES:,} updates",
    )

    # --------------------------------------------------------
    # Scheduler function
    # --------------------------------------------------------

    def lr_factor(step):
        if step < warmup_updates:
            return (
                (step + 1)
                / warmup_updates
            )

        progress = (
            (step - warmup_updates)
            / max(
                1,
                total_updates
                - warmup_updates,
            )
        )

        progress = min(
            1.0,
            progress,
        )

        return (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * progress
                )
            )
        )

    # --------------------------------------------------------
    # Resume discovery
    # --------------------------------------------------------

    latest_checkpoint = (
        find_latest_checkpoint()
    )

    resume_state = None

    if latest_checkpoint is not None:
        print()
        print("=" * 80)
        print("RESUME CHECKPOINT FOUND")
        print("=" * 80)

        print()
        print(latest_checkpoint)

        resume_state = (
            load_checkpoint_state(
                latest_checkpoint,
                dev,
            )
        )

        print()
        print(
            "Epoch:",
            resume_state["epoch"],
        )

        print(
            "Batch in epoch:",
            resume_state[
                "batch_in_epoch"
            ],
        )

        print(
            "Global update:",
            resume_state[
                "global_update"
            ],
        )

    # --------------------------------------------------------
    # Model + LoRA adapter
    # --------------------------------------------------------

    print()
    print("Loading frozen base model...")

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            CFG["model_id"]
        )
    )

    base_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # Keep the audited 1024-token Stage 4 experiment while reducing
    # activation memory. The base model remains frozen under LoRA.
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    if latest_checkpoint is not None:
        print()
        print("Loading trainable LoRA adapter from checkpoint...")

        model = PeftModel.from_pretrained(
            base_model,
            latest_checkpoint,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias=LORA_BIAS,
            task_type=TaskType.CAUSAL_LM,
            fan_in_fan_out=True,
        )

        model = get_peft_model(
            base_model,
            lora_config,
        )

    model = model.to(dev)

    # PEFT can wrap the base model, so enforce training-time cache
    # behavior again after adapter construction/loading.
    model.config.use_cache = False

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    trainable_percentage = (
        100.0
        * trainable_parameters
        / max(1, total_parameters)
    )

    print()
    print("LoRA configuration:")
    print("  rank:                    ", LORA_R)
    print("  alpha:                   ", LORA_ALPHA)
    print("  dropout:                 ", LORA_DROPOUT)
    print("  target modules:          ", ", ".join(LORA_TARGET_MODULES))
    print("  bias:                    ", LORA_BIAS)
    print("  use_cache:               ", model.config.use_cache)
    print("  gradient checkpointing:  ", model.is_gradient_checkpointing)
    print("  total parameters:        ", f"{total_parameters:,}")
    print("  trainable parameters:    ", f"{trainable_parameters:,}")
    print("  trainable percentage:    ", f"{trainable_percentage:.4f}%")

    if trainable_parameters <= 0:
        raise RuntimeError("LoRA model has no trainable parameters.")

    empty_device_cache(dev)

    # --------------------------------------------------------
    # Optimizer - train LoRA parameters only
    # --------------------------------------------------------

    trainable_parameter_list = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameter_list,
        lr=CFG["learning_rate"],
        weight_decay=CFG[
            "weight_decay"
        ],
        foreach=False,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .LambdaLR(
            optimizer,
            lr_factor,
        )
    )

    # --------------------------------------------------------
    # Restore or initialize state
    # --------------------------------------------------------

    if resume_state is not None:
        optimizer.load_state_dict(
            resume_state["optimizer"]
        )

        scheduler.load_state_dict(
            resume_state["scheduler"]
        )

        restore_rng_state(
            resume_state.get("rng")
        )

        start_epoch = (
            resume_state["epoch"]
        )

        resume_batch = (
            resume_state[
                "batch_in_epoch"
            ]
        )

        global_update = (
            resume_state[
                "global_update"
            ]
        )

        best_validation_loss = (
            resume_state[
                "best_validation_loss"
            ]
        )

        initial_validation_loss = (
            resume_state[
                "initial_validation_loss"
            ]
        )

        history = list(
            resume_state.get(
                "history",
                [],
            )
        )

        resumed_epoch_loss_sum = (
            resume_state.get(
                "epoch_loss_sum",
                0.0,
            )
        )

        resumed_epoch_loss_batches = (
            resume_state.get(
                "epoch_loss_batches",
                0,
            )
        )

        # If checkpoint was saved exactly at the end
        # of an epoch, begin the next one.
        if resume_batch >= batches_per_epoch:
            start_epoch += 1
            resume_batch = 0
            resumed_epoch_loss_sum = 0.0
            resumed_epoch_loss_batches = 0

        print()
        print("Resume state restored.")

    else:
        start_epoch = 1
        resume_batch = 0
        global_update = 0

        best_validation_loss = float(
            "inf"
        )

        history = []

        resumed_epoch_loss_sum = 0.0
        resumed_epoch_loss_batches = 0

        # ----------------------------------------------------
        # Baseline validation
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("INITIAL VALIDATION")
        print("=" * 80)

        initial_validation_loss = (
            evaluate(
                model,
                validation_loader,
                dev,
                description=(
                    "Initial validation"
                ),
            )
        )

        print()
        print(
            "Initial validation loss:",
            f"{initial_validation_loss:.6f}",
        )

        empty_device_cache(dev)

    # --------------------------------------------------------
    # Already complete?
    # --------------------------------------------------------

    if start_epoch > CFG["epochs"]:
        print()
        print(
            "Training is already complete."
        )

        return

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        start_epoch,
        CFG["epochs"] + 1,
    ):
        print()
        print("=" * 80)
        print(
            f"EPOCH {epoch}/{CFG['epochs']}"
        )
        print("=" * 80)

        train_loader = (
            make_train_loader(
                train_dataset,
                collator,
                epoch,
            )
        )

        model.train()

        if (
            epoch == start_epoch
            and resume_batch > 0
        ):
            epoch_loss_sum = (
                resumed_epoch_loss_sum
            )

            epoch_loss_batches = (
                resumed_epoch_loss_batches
            )

            skip_batches = resume_batch

            print()
            print(
                f"Resuming epoch {epoch} "
                f"after batch "
                f"{skip_batches:,}/"
                f"{len(train_loader):,}"
            )

        else:
            epoch_loss_sum = 0.0
            epoch_loss_batches = 0
            skip_batches = 0

        progress = tqdm(
            total=len(train_loader),
            initial=skip_batches,
            desc=(
                f"Epoch "
                f"{epoch}/{CFG['epochs']}"
            ),
            dynamic_ncols=True,
        )

        accumulation_counter = 0

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            # ------------------------------------------------
            # Resume: deterministic loader order allows us
            # to skip batches already consumed.
            # ------------------------------------------------

            if batch_index <= skip_batches:
                continue

            batch = {
                key: value.to(dev)
                for key, value
                in batch.items()
            }

            output = model(
                **batch
            )

            raw_loss = output.loss

            require_finite_loss(
                raw_loss,
                phase=f"epoch {epoch} training",
                batch_index=batch_index,
            )

            raw_loss_value = float(
                raw_loss.detach().cpu()
            )

            epoch_loss_sum += (
                raw_loss_value
            )

            epoch_loss_batches += 1

            loss = (
                raw_loss
                / accumulation_steps
            )

            loss.backward()

            accumulation_counter += 1

            should_update = (
                accumulation_counter
                >= accumulation_steps
                or batch_index
                == len(train_loader)
            )

            if should_update:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameter_list,
                    CFG["max_grad_norm"],
                )

                if not torch.isfinite(grad_norm).item():
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(
                        f"Non-finite gradient norm at epoch {epoch}, "
                        f"batch {batch_index}: "
                        f"{float(grad_norm.detach().cpu())}"
                    )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                accumulation_counter = 0
                global_update += 1

            current_lr = (
                scheduler.get_last_lr()[0]
            )

            mean_loss = (
                epoch_loss_sum
                / max(
                    1,
                    epoch_loss_batches,
                )
            )

            progress.update(1)

            progress.set_postfix(
                loss=(
                    f"{raw_loss_value:.4f}"
                ),
                avg=(
                    f"{mean_loss:.4f}"
                ),
                lr=(
                    f"{current_lr:.2e}"
                ),
                update=(
                    f"{global_update}/"
                    f"{total_updates}"
                ),
                refresh=False,
            )

            # ------------------------------------------------
            # Periodic full checkpoint.
            #
            # Only save immediately after an optimizer update,
            # never in the middle of gradient accumulation.
            # ------------------------------------------------

            if (
                should_update
                and global_update > 0
                and (
                    global_update
                    % CHECKPOINT_EVERY_UPDATES
                    == 0
                )
            ):
                progress.refresh()

                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    batch_in_epoch=(
                        batch_index
                    ),
                    global_update=(
                        global_update
                    ),
                    total_updates=(
                        total_updates
                    ),
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    initial_validation_loss=(
                        initial_validation_loss
                    ),
                    history=history,
                    epoch_loss_sum=(
                        epoch_loss_sum
                    ),
                    epoch_loss_batches=(
                        epoch_loss_batches
                    ),
                )

                write_metrics(
                    initial_validation_loss,
                    best_validation_loss,
                    global_update,
                    total_updates,
                    history,
                    status="training",
                )

        progress.close()

        # ----------------------------------------------------
        # Epoch validation
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch} training "
            f"mean loss: "
            f"{epoch_loss_sum / max(1, epoch_loss_batches):.6f}"
        )

        validation_loss = (
            evaluate(
                model,
                validation_loader,
                dev,
                description=(
                    f"Validation "
                    f"{epoch}/{CFG['epochs']}"
                ),
            )
        )

        empty_device_cache(dev)

        epoch_record = {
            "epoch":
                epoch,

            "update":
                global_update,

            "training_loss":
                (
                    epoch_loss_sum
                    / max(
                        1,
                        epoch_loss_batches,
                    )
                ),

            "validation_loss":
                validation_loss,

            "learning_rate":
                scheduler.get_last_lr()[0],
        }

        history.append(
            epoch_record
        )

        print()
        print(
            json.dumps(
                epoch_record,
                indent=2,
            )
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if (
            validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_loss
            )

            print()
            print(
                "New best validation loss:",
                f"{best_validation_loss:.6f}",
            )

            remove_directory(
                BEST_DIR
            )

            BEST_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                BEST_DIR,
                safe_serialization=True,
            )

            tokenizer.save_pretrained(
                BEST_DIR
            )

        # ----------------------------------------------------
        # Always save a resumable checkpoint at epoch end.
        # ----------------------------------------------------

        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            batch_in_epoch=(
                len(train_loader)
            ),
            global_update=(
                global_update
            ),
            total_updates=(
                total_updates
            ),
            best_validation_loss=(
                best_validation_loss
            ),
            initial_validation_loss=(
                initial_validation_loss
            ),
            history=history,
            epoch_loss_sum=(
                epoch_loss_sum
            ),
            epoch_loss_batches=(
                epoch_loss_batches
            ),
        )

        write_metrics(
            initial_validation_loss,
            best_validation_loss,
            global_update,
            total_updates,
            history,
            status="training",
        )

        # Resume information applies only to first resumed epoch.
        resume_batch = 0
        resumed_epoch_loss_sum = 0.0
        resumed_epoch_loss_batches = 0

    # ========================================================
    # FINAL MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)

    remove_directory(
        FINAL_DIR
    )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        FINAL_DIR,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        FINAL_DIR
    )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    write_metrics(
        initial_validation_loss,
        best_validation_loss,
        global_update,
        total_updates,
        history,
        status="complete",
    )

    final_metrics = {
        "initial_validation_loss":
            initial_validation_loss,

        "best_validation_loss":
            best_validation_loss,

        "optimizer_updates":
            global_update,

        "total_optimizer_updates":
            total_updates,

        "history":
            history,

        "lora_config":
            lora_experiment_config(),
    }

    print()
    print("=" * 80)
    print("STAGE 4 LoRA TRAINING COMPLETE")
    print("=" * 80)

    print()
    print(
        json.dumps(
            final_metrics,
            indent=2,
        )
    )

    print()
    print(
        "Best model:",
        BEST_DIR,
    )

    print(
        "Final model:",
        FINAL_DIR,
    )

    print(
        "Metrics:",
        METRICS_FILE,
    )


if __name__ == "__main__":
    main()