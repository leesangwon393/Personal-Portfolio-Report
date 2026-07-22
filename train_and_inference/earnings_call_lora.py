from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from secrets_config import get_secret


API_URL = "https://www.alphavantage.co/query"
QUARTERS = ("Q1", "Q2", "Q3", "Q4")


def executive_turns(payload: dict) -> list[dict]:
    ticker = str(payload.get("symbol", "")).upper()
    quarter = str(payload.get("quarter", ""))
    rows = []
    for turn in payload.get("transcript", []):
        title = str(turn.get("title", ""))
        content = str(turn.get("content", "")).strip()
        if "chief executive officer" not in title.lower() and "chief financial officer" not in title.lower():
            continue
        if content:
            rows.append(
                {
                    "ticker": ticker,
                    "quarter": quarter,
                    "speaker": str(turn.get("speaker", "")).strip(),
                    "title": title,
                    "text": content,
                }
            )
    return rows


def collect(args: argparse.Namespace) -> None:
    api_key = get_secret("ALPHAVANTAGE_API_KEY", required=True)
    tickers = pd.read_csv(args.tickers_csv)["Ticker"].dropna().astype(str).str.upper().tolist()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for ticker in dict.fromkeys(tickers):
            for suffix in QUARTERS:
                quarter = f"{args.year}{suffix}"
                response = requests.get(
                    API_URL,
                    params={
                        "function": "EARNINGS_CALL_TRANSCRIPT",
                        "symbol": ticker,
                        "quarter": quarter,
                        "apikey": api_key,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                if "transcript" not in payload:
                    print(f"skip {ticker} {quarter}: {payload.get('Information') or payload.get('Note') or 'no transcript'}")
                    continue
                for row in executive_turns(payload):
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"saved {ticker} {quarter}")


def train(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    token = get_secret("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, token=token)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CEO/CFO earnings-call turns and train a LoRA adapter.")
    commands = parser.add_subparsers(dest="command", required=True)

    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--tickers-csv", default="data/S&P500_metrics.csv")
    collect_parser.add_argument("--year", type=int, default=2025)
    collect_parser.add_argument("--output", default="local_data/earnings_calls/ceo_cfo_2025.jsonl")
    collect_parser.set_defaults(func=collect)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--dataset", default="local_data/earnings_calls/ceo_cfo_2025.jsonl")
    train_parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    train_parser.add_argument("--output", default="local_data/earnings_calls/lora_adapter")
    train_parser.add_argument("--max-length", type=int, default=1024)
    train_parser.add_argument("--epochs", type=float, default=1)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
