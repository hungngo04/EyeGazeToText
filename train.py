import argparse
import ast

import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Keyboard regions grouped by spatial proximity on a QWERTY layout
CLUSTER_MAPS = {
    9: {"qwe": "1", "rtyu": "2", "iop": "3", "asd": "4", "fgh": "5", "jkl": "6", "zxcv": "7", "bnm": "9"},
    7: {"qwert": "1", "yuiop": "2", "asdfg": "3", "hjkl": "4", "zxc": "5", "vbnm": "6"},
    5: {"qweasdz": "1", "rtfgxc": "2", "yuhjvb": "3", "iopklnm": "4"},
    3: {"qwertasdfgzxc": "1", "yuiophjklvbnm": "2"},
}

PREFIX = "translate Cluster to English: "


def build_char_map(num_clusters):
    char_map = {}
    for keys, digit in CLUSTER_MAPS[num_clusters].items():
        for ch in keys:
            char_map[ch] = digit
    return char_map


def convert_to_cluster_sequence(text, num_clusters):
    char_map = build_char_map(num_clusters)
    result = []
    for c in text.lower():
        if c.isalpha():
            result.append(char_map.get(c, ""))
        elif c == " ":
            result.append(" ")
    return "".join(result)


def prepare_dataset(input_csv, output_csv, num_clusters):
    df = pd.read_csv(input_csv, delimiter="\t", usecols=[0], names=["text", "fr"])
    df["text"] = df["text"].astype(str).str.lower()
    df = df.drop_duplicates(subset="text")
    df["cluster"] = df["text"].apply(lambda t: convert_to_cluster_sequence(t, num_clusters))

    dataset = []
    for i, row in df.iterrows():
        dataset.append({
            "id": str(i),
            "translation": str({
                "cluster": row["cluster"].replace(",", ""),
                "en": row["text"].replace(",", ""),
            }),
        })

    out = pd.DataFrame(dataset)
    out.to_csv(output_csv, header=True, index=False)
    print(f"Saved {len(out)} examples to {output_csv}")


def preprocess_function(examples, tokenizer):
    inputs = [PREFIX + ast.literal_eval(ex)["cluster"] for ex in examples["translation"]]
    targets = [ast.literal_eval(ex)["en"] for ex in examples["translation"]]
    return tokenizer(inputs, text_target=targets, max_length=256, truncation=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def make_compute_metrics(tokenizer):
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def train(args):
    dataset = load_dataset("csv", data_files=args.dataset)
    dataset = dataset["train"].train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = dataset.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    trainer.train()
    if args.push_to_hub:
        trainer.push_to_hub()


def infer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    text = PREFIX + args.text
    inputs = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser(description="Touchless Typing Trainer - cluster sequence to English translation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Convert English text to cluster sequences")
    prep.add_argument("--input", default="eng-fra-original.csv")
    prep.add_argument("--output", default="translation-cluster-eng-dataset.csv")
    prep.add_argument("--clusters", type=int, default=9, choices=[3, 5, 7, 9])

    tr = subparsers.add_parser("train", help="Fine-tune a seq2seq model")
    tr.add_argument("--dataset", default="translation-cluster-eng-dataset.csv")
    tr.add_argument("--model", default="facebook/bart-large")
    tr.add_argument("--output-dir", default="cluster_to_text_model")
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--learning-rate", type=float, default=2e-5)
    tr.add_argument("--fp16", action="store_true")
    tr.add_argument("--push-to-hub", action="store_true")

    inf = subparsers.add_parser("infer", help="Translate a cluster sequence to English")
    inf.add_argument("--model", required=True)
    inf.add_argument("--text", required=True, help="Cluster sequence, e.g. '51663 531 421'")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_dataset(args.input, args.output, args.clusters)
    elif args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)


if __name__ == "__main__":
    main()
