#!/usr/bin/env python
"""Unsloth QLoRA fine-tune of Gemma 4 E4B on antibody language.

Day-14 (GPU phase) script. Runs on GCP A100 40GB spot (~5 hrs, ~$5.50).

Training data:
    OAS healthy human heavy chains (FASTA from `scripts/download_oas.py`).
    Each sequence is wrapped in <antibody>...</antibody> sentinels so the
    model learns to distinguish antibody language from generic text.

Held-out 5 % is used for perplexity evaluation. Target: held-out perplexity
≥ 30 % lower than the base model.

Output:
    runs/<out_dir>/lora/                — adapter weights + tokenizer
    runs/<out_dir>/metrics.json         — train/eval loss, perplexity, n_*

If --push-to-hub and HF_TOKEN are set, the adapter is pushed to
huggingface.co/<hub-id>.

Usage:
    python scripts/finetune_gemma4_unsloth.py \\
        --base-model unsloth/gemma-4-e4b \\
        --train-fasta data/oas/oas_paired.fasta \\
        --out runs/gemma4-e4b-ab-lora \\
        --epochs 1 --lr 2e-4 \\
        --push-to-hub --hub-id YOUR_USER/idiotypeforge-gemma4-e4b-ab-lora
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click


@click.command()
@click.option("--base-model", default="unsloth/gemma-4-e4b",
              help="Base model on Hugging Face Hub.")
@click.option("--train-fasta", type=click.Path(exists=True), required=True,
              help="OAS FASTA (one sequence per record).")
@click.option("--out", "out_dir", type=click.Path(file_okay=False),
              default="runs/gemma4-e4b-ab-lora")
@click.option("--epochs", type=int, default=1)
@click.option("--lr", type=float, default=2e-4)
@click.option("--seq-len", type=int, default=1024)
# L4-tuned defaults: batch_size=8 fits comfortably in 24 GB at seq_len=1024
# with 4-bit base + bf16 LoRA + FA2 + grad checkpointing. grad_accum=2 keeps
# effective batch at 16 — same effective LR as the previous (4 × 4) config.
@click.option("--batch-size", type=int, default=8,
              help="Per-device train batch size. 8 = L4 24GB sweet spot; 16 = A100 40GB.")
@click.option("--grad-accum", type=int, default=2)
@click.option("--lora-r", type=int, default=16)
@click.option("--lora-alpha", type=int, default=32)
# Packing concatenates multiple short antibody sequences (~110 aa each) into
# one seq_len-token block per training example. With seq_len=1024 we fit
# ~9 antibodies per example, giving ~9× sample throughput vs naive padding.
@click.option("--packing/--no-packing", default=True,
              help="Pack multiple short antibody seqs into one seq_len block (~9× throughput).")
@click.option("--num-workers", type=int, default=4,
              help="DataLoader workers. 4 keeps the GPU fed on L4.")
@click.option("--max-sequences", type=int, default=200_000,
              help="Cap on training sequences (200K = ~3-5 hrs A100, ~6 hrs L4).")
@click.option("--push-to-hub/--no-push", default=False)
@click.option("--hub-id", default=None,
              help="HF Hub repo id (e.g. user/idiotypeforge-gemma4-e4b-ab-lora).")
@click.option("--benchmark-only", is_flag=True, default=False,
              help="Skip training; only run the CDR3-masked-AA top-1 accuracy "
                   "benchmark on the (optionally LoRA-adapted) base model.")
@click.option("--lora-adapter", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Path to a saved LoRA adapter dir. When set with "
                   "--benchmark-only, evaluates the fine-tuned model.")
@click.option("--n-bench-seqs", type=int, default=500,
              help="How many held-out sequences to score in the CDR3 benchmark.")
def main(
    base_model: str,
    train_fasta: str,
    out_dir: str,
    epochs: int,
    lr: float,
    seq_len: int,
    batch_size: int,
    grad_accum: int,
    lora_r: int,
    lora_alpha: int,
    packing: bool,
    num_workers: int,
    max_sequences: int,
    push_to_hub: bool,
    hub_id: str | None,
    benchmark_only: bool,
    lora_adapter: str | None,
    n_bench_seqs: int,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo(f"→ Base model: {base_model}")
    click.echo(f"→ Output:     {out}")

    # Lazy imports — heavy GPU deps. The script can be inspected/imported on
    # CPU without touching them.
    try:
        import torch                       # type: ignore[import-not-found]
        from datasets import Dataset       # type: ignore[import-not-found]
        from trl import SFTConfig, SFTTrainer    # type: ignore[import-not-found]
        from unsloth import FastLanguageModel    # type: ignore[import-not-found]
    except ImportError as e:
        click.echo(f"✗ Missing GPU deps: {e}\n  Run scripts/setup_a100.sh on the VM.", err=True)
        sys.exit(1)

    if not torch.cuda.is_available():
        click.echo("✗ CUDA unavailable. Fine-tune requires a GPU.", err=True)
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 1. Load OAS sequences from FASTA
    # ---------------------------------------------------------------------
    click.echo(f"→ Loading sequences from {train_fasta}")
    sequences: list[str] = []
    cur: list[str] | None = None
    with Path(train_fasta).open() as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None and cur:
                    sequences.append("".join(cur))
                cur = []
                if len(sequences) >= max_sequences:
                    break
            elif cur is not None:
                cur.append(line.strip())
        if cur is not None and cur:
            sequences.append("".join(cur))

    sequences = [s for s in sequences if 50 < len(s) < seq_len][:max_sequences]
    click.echo(f"  · loaded {len(sequences):,} sequences (after length filter)")

    if not sequences:
        click.echo("✗ No usable sequences. Did download_oas.py run successfully?", err=True)
        sys.exit(1)

    # Wrap each sequence in <antibody> sentinels — gives the model a clear
    # signal that this is antibody language vs. arbitrary protein text.
    # The "raw" field is consumed by the CDR3 masked-AA benchmark (it needs
    # the unwrapped sequence for ANARCI numbering); SFTTrainer's
    # `dataset_text_field="text"` ignores the extra column.
    formatted = [
        {"text": f"<antibody>\n{s}\n</antibody>", "raw": s} for s in sequences
    ]

    # 95/5 split
    n_eval = max(200, len(formatted) // 20)
    train_records = formatted[:-n_eval]
    eval_records = formatted[-n_eval:]
    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)
    click.echo(f"  · train={len(train_ds):,}, eval={len(eval_ds):,}")

    # ---------------------------------------------------------------------
    # 2. Load model with Unsloth 4-bit + apply LoRA
    # ---------------------------------------------------------------------
    click.echo(f"→ Loading {base_model} in 4-bit (Unsloth)…")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=seq_len,
        load_in_4bit=True,
        dtype=None,
    )

    if benchmark_only:
        # Benchmark-only path: optionally load a saved LoRA adapter on top of
        # the base, run the CDR3 masked-AA top-1 benchmark, write
        # baseline_metrics.json, and exit. Used to capture both the base
        # baseline (~25%) and the fine-tuned number (≥50%) in separate runs.
        if lora_adapter:
            from peft import PeftModel              # type: ignore[import-not-found]
            click.echo(f"→ Loading LoRA adapter from {lora_adapter}")
            model = PeftModel.from_pretrained(model, lora_adapter)
        FastLanguageModel.for_inference(model)
        from app.eval.cdr3_masked import compute_cdr3_masked_top1
        click.echo(f"→ Running CDR3-masked-AA benchmark on {n_bench_seqs} sequences…")
        bench = compute_cdr3_masked_top1(
            model=model, tokenizer=tokenizer, eval_records=eval_records,
            n_seqs=n_bench_seqs,
        )
        bench_payload = {
            "base_model": base_model,
            "lora_adapter": lora_adapter,
            **bench,
        }
        (out / "baseline_metrics.json").write_text(json.dumps(bench_payload, indent=2))
        click.echo(f"  · {bench_payload}")
        click.echo(f"✓ Benchmark-only complete: {out / 'baseline_metrics.json'}")
        return

    click.echo(f"→ Attaching LoRA (r={lora_r}, alpha={lora_alpha})…")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ---------------------------------------------------------------------
    # 3. Train
    # ---------------------------------------------------------------------
    click.echo("→ Starting training…")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=seq_len,
        # Packing concatenates multiple short antibody sequences (~110 aa)
        # per training example up to seq_len, ~9× sample throughput.
        packing=packing,
        args=SFTConfig(
            output_dir=str(out / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=20,
            save_steps=500,
            save_total_limit=2,
            eval_steps=500,
            # Mixed precision — Unsloth picks the right one based on GPU caps;
            # L4 / A100 both support BF16, T4 falls back to FP16.
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            seed=42,
            report_to=[],
            # Keep the GPU fed: 4 dataloader workers, pin memory so transfers
            # overlap with compute. group_by_length avoids padding waste when
            # packing is off.
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            group_by_length=not packing,
        ),
    )
    train_result = trainer.train()

    # ---------------------------------------------------------------------
    # 4. Eval
    # ---------------------------------------------------------------------
    click.echo("→ Evaluating perplexity on held-out set…")
    eval_result = trainer.evaluate()
    eval_loss = float(eval_result.get("eval_loss", float("nan")))
    perplexity = float(torch.exp(torch.tensor(eval_loss)).item()) if eval_loss == eval_loss else float("nan")

    # Inline CDR3-masked-AA top-1 benchmark — pairs naturally with perplexity
    # to claim the plan's "≥50% vs ~25% base" threshold. Runs ~5 min on A100
    # at n_bench_seqs=500.
    click.echo(f"→ Running CDR3-masked-AA benchmark on {n_bench_seqs} sequences…")
    FastLanguageModel.for_inference(model)
    from app.eval.cdr3_masked import compute_cdr3_masked_top1
    bench = compute_cdr3_masked_top1(
        model=model, tokenizer=tokenizer, eval_records=eval_records,
        n_seqs=n_bench_seqs,
    )

    metrics = {
        "base_model": base_model,
        "epochs": epochs,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "lora_rank": lora_r,
        "lora_alpha": lora_alpha,
        "seq_len": seq_len,
        "cdr3_masked_top1_accuracy": bench["top1_accuracy"],
        "cdr3_masked_n_positions": bench["n_positions"],
        "cdr3_masked_n_anarci_failures": bench["n_anarci_failures"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    click.echo(f"  · {metrics}")

    # ---------------------------------------------------------------------
    # 5. Save adapter + push to Hub
    # ---------------------------------------------------------------------
    adapter_dir = out / "lora"
    click.echo(f"→ Saving LoRA adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if push_to_hub:
        if not hub_id:
            click.echo("✗ --push-to-hub set but --hub-id missing.", err=True)
            sys.exit(1)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            click.echo("✗ HF_TOKEN env var required to push to Hub.", err=True)
            sys.exit(1)
        click.echo(f"→ Pushing adapter to huggingface.co/{hub_id}")
        model.push_to_hub(hub_id, token=token)
        tokenizer.push_to_hub(hub_id, token=token)

    click.echo(f"✓ Done. Adapter: {adapter_dir}  Metrics: {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
