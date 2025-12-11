"""
Training script for Cross-Encoder (Reranker) fine-tuning
Using CrossEncoderTrainer API (similar to training_ms_marco_bce.py)
Adapted for ADL HW3 dataset format
"""

import logging
import json
import random
import torch
from pathlib import Path
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.training_args import BatchSamplers
import argparse

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

def load_corpus(corpus_path):
    """Load corpus and create pid to text mapping"""
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            corpus[data['id']] = data['text']
    logging.info(f"Loaded {len(corpus)} passages from corpus")
    return corpus

def load_qrels(qrels_path):
    """Load qrels (query to positive passage mapping)"""
    with open(qrels_path, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    logging.info(f"Loaded qrels for {len(qrels)} queries")
    return qrels

def create_dataset(train_path, qrels, corpus, num_neg_per_query=4, max_samples=None):
    """
    Create Hugging Face Dataset for CrossEncoderTrainer
    
    Returns:
        Dataset with columns: ['query', 'passage', 'label']
    """
    queries = []
    passages = []
    labels = []
    skipped = 0
    
    with open(train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if max_samples:
            lines = lines[:max_samples]
        
        for line in lines:
            data = json.loads(line)
            qid = data['qid']
            query = data['rewrite']
            evidences = data['evidences']
            evidence_labels = data['retrieval_labels']
            
            # Get positive passage from qrels
            if qid not in qrels:
                skipped += 1
                continue
            
            positive_pids = list(qrels[qid].keys())
            if not positive_pids:
                skipped += 1
                continue
            
            positive_pid = positive_pids[0]
            
            if positive_pid not in corpus:
                skipped += 1
                continue
            
            positive_text = corpus[positive_pid]
            
            # Add positive sample
            queries.append(query)
            passages.append(positive_text)
            labels.append(1)
            
            # Collect negative samples from evidences (BM25 hard negatives)
            negative_samples = []
            for evidence, label in zip(evidences, evidence_labels):
                if label == 0:
                    negative_samples.append(evidence)
            
            # If not enough negatives, sample randomly from corpus
            if len(negative_samples) < num_neg_per_query:
                corpus_texts = list(corpus.values())
                num_to_sample = num_neg_per_query - len(negative_samples)
                
                for _ in range(num_to_sample * 10):
                    random_text = random.choice(corpus_texts)
                    if random_text != positive_text and random_text not in negative_samples:
                        negative_samples.append(random_text)
                        if len(negative_samples) >= num_neg_per_query:
                            break
            
            # Add negative samples
            for neg_text in negative_samples[:num_neg_per_query]:
                queries.append(query)
                passages.append(neg_text)
                labels.append(0)
    
    # Create Hugging Face Dataset
    dataset = Dataset.from_dict({
        'query': queries,
        'passage': passages,
        'label': labels
    })
    
    logging.info(f"Created dataset with {len(dataset)} samples ({skipped} queries skipped)")
    logging.info(f"  Positive samples: {sum(labels)}")
    logging.info(f"  Negative samples: {len(labels) - sum(labels)}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--corpus_path", type=str, default="./data/corpus.txt")
    parser.add_argument("--train_path", type=str, default="./data/train.txt")
    parser.add_argument("--qrels_path", type=str, default="./data/qrels.txt")
    parser.add_argument("--output_path", type=str, default="./models/reranker")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_neg_per_query", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_split", type=float, default=0.05, help="Fraction of data for evaluation")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", action='store_true', help="Use FP16 training")
    parser.add_argument("--bf16", action='store_true', help="Use BF16 training")
    args = parser.parse_args()
    
    # Load data
    logging.info("="*60)
    logging.info("Loading data...")
    corpus = load_corpus(args.corpus_path)
    qrels = load_qrels(args.qrels_path)
    
    logging.info("Creating dataset...")
    dataset = create_dataset(
        args.train_path,
        qrels,
        corpus,
        num_neg_per_query=args.num_neg_per_query,
        max_samples=args.max_samples
    )
    
    # Split into train and eval
    logging.info(f"Splitting dataset (eval ratio: {args.eval_split})...")
    dataset = dataset.train_test_split(test_size=args.eval_split, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    logging.info(f"Train dataset: {len(train_dataset)} samples")
    logging.info(f"Eval dataset: {len(eval_dataset)} samples")
    
    # Training configuration
    logging.info("="*60)
    logging.info("Training Configuration:")
    logging.info(f"  Model: {args.model_name}")
    logging.info(f"  Train samples: {len(train_dataset)}")
    logging.info(f"  Eval samples: {len(eval_dataset)}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Epochs: {args.num_epochs}")
    logging.info(f"  Learning rate: {args.learning_rate}")
    logging.info(f"  Warmup ratio: {args.warmup_ratio}")
    logging.info(f"  Negatives per query: {args.num_neg_per_query}")
    logging.info(f"  FP16: {args.fp16}, BF16: {args.bf16}")
    logging.info(f"  Output path: {args.output_path}")
    logging.info("="*60)
    
    # 1. Define CrossEncoder model
    logging.info("Loading CrossEncoder model...")
    torch.manual_seed(42)
    model = CrossEncoder(args.model_name, num_labels=1, max_length=512)
    logging.info(f"Model max length: {model.max_length}")
    logging.info(f"Model num labels: {model.num_labels}")
    
    # 2. Define loss function
    loss = BinaryCrossEntropyLoss(model)
    
    # 3. Define training arguments
    training_args = CrossEncoderTrainingArguments(
        # Required parameter
        output_dir=args.output_path,
        
        # Training hyperparameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        
        # Mixed precision
        fp16=args.fp16,
        bf16=args.bf16,
        
        # Batch sampler
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        
        # Evaluation & checkpointing
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        
        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        
        # Other
        seed=42,
        report_to=[],  # Disable wandb/tensorboard
    )
    
    # 4. Create trainer
    logging.info("Creating CrossEncoderTrainer...")
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    
    # 5. Start training
    logging.info("="*60)
    logging.info("Starting training...")
    logging.info("="*60)
    trainer.train()
    
    # 6. Save final model
    logging.info("="*60)
    logging.info("Training completed!")
    logging.info(f"Saving final model to {args.output_path}/final")
    
    final_output_dir = f"{args.output_path}/final"
    model.save_pretrained(final_output_dir)
    
    logging.info("="*60)
    logging.info("Training completed successfully!")
    logging.info(f"Model checkpoints saved to: {args.output_path}")
    logging.info(f"Final model saved to: {final_output_dir}")
    logging.info("="*60)
    
    # Print file structure
    logging.info("\nSaved files:")
    output_path = Path(args.output_path)
    for file in sorted(output_path.rglob("*")):
        if file.is_file():
            size_mb = file.stat().st_size / 1024 / 1024
            logging.info(f"  {file.relative_to(output_path)} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
