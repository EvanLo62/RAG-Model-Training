"""
修正後的 Retriever 訓練腳本
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, evaluation

# 設定 logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO, 
    handlers=[LoggingHandler()]
)

# 參數設定
parser = argparse.ArgumentParser(description='Retriever 訓練')
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--model_name", default="intfloat/multilingual-e5-small", type=str)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negatives", default=4, type=int, help="每個 query 的隨機負樣本數量")
parser.add_argument("--evaluation_steps", default=0, type=int)
parser.add_argument("--log_steps", default=50, type=int)

# 資料路徑
parser.add_argument("--train_path", default="./data/train.txt", type=str)
parser.add_argument("--corpus_path", default="./data/corpus.txt", type=str)
parser.add_argument("--qrels_path", default="./data/qrels.txt", type=str)
parser.add_argument("--output_path", default="./models/retriever", type=str)

args = parser.parse_args()

print("=" * 70)
print("訓練參數:")
print("=" * 70)
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print("=" * 70)

# 載入預訓練模型
logging.info(f"載入模型: {args.model_name}")
model = SentenceTransformer(args.model_name)
model.max_seq_length = args.max_seq_length

# ============================================================================
# 1. 載入資料
# ============================================================================

def load_corpus(corpus_path: str) -> Dict[str, str]:
    """
    載入 corpus

    Returns:
        pid_to_text: pid -> text 的字典
    """
    pid_to_text = {}
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']  # 只使用text, 不處理title
            
            # 加上 E5 格式的 prefix
            text = "passage: " + text
            pid = data['id']
            
            pid_to_text[pid] = text
    
    logging.info(f"載入 {len(pid_to_text)} 篇文章")
    return pid_to_text


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """載入 qrels"""
    with open(qrels_path, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    logging.info(f"載入 {len(qrels)} 個 queries 的 qrels")
    return qrels


def load_train_data(train_path: str) -> List[Dict]:
    """載入訓練資料"""
    data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    logging.info(f"載入 {len(data)} 筆訓練資料")
    return data


# 載入所有資料
logging.info("=" * 70)
logging.info("開始載入資料")
logging.info("=" * 70)

corpus = load_corpus(args.corpus_path)
qrels = load_qrels(args.qrels_path)
train_data = load_train_data(args.train_path)

# ============================================================================
# 2. 準備訓練資料
# ============================================================================

def prepare_train_queries(
    train_data: List[Dict],
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    num_negatives: int = 4
) -> Dict[str, Dict]:
    """
    準備訓練用的 queries
    
    策略:
    - 使用隨機負樣本 (避免 evidences 的標註問題)
    - 依賴 MNRL 的 in-batch negatives (batch_size - 1 個額外負樣本)
    - 總負樣本數 = num_negatives (顯式) + (batch_size - 1) (in-batch)
    
    """
    train_queries = {}
    corpus_pids = list(corpus.keys())
    
    stats = {
        'total': 0,
        'with_pos_and_neg': 0,
        'skipped_no_pos': 0,
    }
    
    for item in tqdm(train_data, desc="準備訓練 queries"):
        qid = item['qid']
        query = item['rewrite']
        
        # 加上 query prefix (E5 格式)
        query = "query: " + query
        
        # 1. 從 qrels 取得 positive passages
        if qid not in qrels:
            stats['skipped_no_pos'] += 1
            continue
        
        pos_pids = [pid for pid, label in qrels[qid].items() if str(label) != "0"]
        if len(pos_pids) == 0:
            stats['skipped_no_pos'] += 1
            continue
        
        # 確保所有 positive pids 都在 corpus 中
        pos_pids = [pid for pid in pos_pids if pid in corpus]
        if len(pos_pids) == 0:
            stats['skipped_no_pos'] += 1
            continue
        
        # 2. 隨機選擇負樣本
        neg_pids_list = []
        max_attempts = 1000
        attempts = 0
        
        while len(neg_pids_list) < num_negatives and attempts < max_attempts:
            pid = random.choice(corpus_pids)
            if pid not in pos_pids and pid not in neg_pids_list:
                neg_pids_list.append(pid)
            attempts += 1
        
        # 記錄
        if len(pos_pids) > 0 and len(neg_pids_list) > 0:
            train_queries[qid] = {
                'qid': qid,
                'query': query,
                'pos': pos_pids,
                'neg': neg_pids_list,
            }
            stats['with_pos_and_neg'] += 1
        
        stats['total'] += 1
    
    # 輸出統計
    logging.info("=" * 70)
    logging.info("訓練資料統計:")
    logging.info("=" * 70)
    logging.info(f"總 queries: {stats['total']}")
    logging.info(f"有效 queries: {stats['with_pos_and_neg']}")
    logging.info(f"跳過 (無正樣本): {stats['skipped_no_pos']}")
    logging.info(f"每個 query 的顯式負樣本: {num_negatives}")
    
    return train_queries


train_queries = prepare_train_queries(
    train_data, 
    corpus, 
    qrels, 
    num_negatives=args.num_negatives
)

# ============================================================================
# 3. 建立 Dataset
# ============================================================================

class RetrievalDataset(Dataset):
    """修正版 Dataset,確保所有 pid 都能正確查找"""
    
    def __init__(self, queries: Dict, corpus: Dict):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        
        # 轉為 list 並打亂順序
        for qid in self.queries:
            self.queries[qid]["pos"] = list(self.queries[qid]["pos"])
            self.queries[qid]["neg"] = list(self.queries[qid]["neg"])
            random.shuffle(self.queries[qid]["neg"])
    
    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query["query"]
        
        # 循環使用 positive
        pos_id = query["pos"].pop(0)
        pos_text = self.corpus[pos_id]  # 直接取得,不應該找不到
        query["pos"].append(pos_id)
        
        # 循環使用 negative
        neg_id = query["neg"].pop(0)
        neg_text = self.corpus[neg_id]  # 直接取得,不應該找不到
        query["neg"].append(neg_id)
        
        return InputExample(texts=[query_text, pos_text, neg_text])
    
    def __len__(self):
        return len(self.queries)


# 建立 Dataset 和 DataLoader
logging.info("=" * 70)
logging.info("建立 Dataset 和 DataLoader")
logging.info("=" * 70)

train_dataset = RetrievalDataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(
    train_dataset, 
    shuffle=True, 
    batch_size=args.train_batch_size
)

logging.info(f"Dataset 大小: {len(train_dataset)}")
logging.info(f"Batch size: {args.train_batch_size}")
logging.info(f"每 epoch steps: {len(train_dataloader)}")
logging.info(f"總 steps: {len(train_dataloader) * args.epochs}")

# ============================================================================
# 4. 設定 Loss Function
# ============================================================================

train_loss = losses.MultipleNegativesRankingLoss(model=model)

logging.info("=" * 70)
logging.info("Loss Function: MultipleNegativesRankingLoss")
logging.info("=" * 70)
logging.info(f"每個 query 的負樣本:")
logging.info(f"  - 顯式負樣本 (隨機): {args.num_negatives}")
logging.info(f"  - In-batch negatives: {args.train_batch_size - 1}")
logging.info(f"  - 總計: ~{args.num_negatives + args.train_batch_size - 1}")
logging.info(f"\nMNRL 策略:")
logging.info(f"  - 使用 cosine similarity 作為相似度度量")
logging.info(f"  - 同一 batch 內的其他正樣本自動成為負樣本")
logging.info(f"  - 較大的 batch_size 提供更多 in-batch negatives")

# ============================================================================
# 5. 訓練記錄器
# ============================================================================

class TrainingLogger:
    def __init__(self, output_path: str, log_steps: int = 50):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_path / "training_log.json"
        self.log_steps = log_steps
        self.logs = []
        self.current_step = 0
        
    def __call__(self, score, epoch, steps):
        self.current_step = steps
        
        if steps % self.log_steps == 0 or steps == 0:
            log_entry = {
                'step': steps,
                'epoch': epoch,
                'loss': score if isinstance(score, (int, float)) else None,
                'timestamp': datetime.now().isoformat()
            }
            self.logs.append(log_entry)
            self.save()
            
    def save(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'logs': self.logs,
                'config': vars(args)
            }, f, indent=2, ensure_ascii=False)


training_logger = TrainingLogger(args.output_path, log_steps=args.log_steps)

# ============================================================================
# 6. 準備評估資料
# ============================================================================

evaluator = None
if args.evaluation_steps > 0:
    logging.info("=" * 70)
    logging.info("準備評估資料")
    logging.info("=" * 70)
    
    eval_queries = {}
    eval_corpus = corpus
    eval_relevant_docs = {}
    
    eval_qids = random.sample(list(train_queries.keys()), min(500, len(train_queries)))
    
    for qid in eval_qids:
        query_info = train_queries[qid]
        eval_queries[qid] = query_info['query']
        eval_relevant_docs[qid] = set(query_info['pos'])
    
    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant_docs,
        name='train-eval',
        show_progress_bar=True,
        write_csv=True,
    )
    
    logging.info(f" 評估資料準備完成")
    logging.info(f" 評估 queries: {len(eval_queries)}")

# ============================================================================
# 7. 開始訓練
# ============================================================================

logging.info("=" * 70)
logging.info("開始訓練")
logging.info("=" * 70)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    evaluation_steps=args.evaluation_steps if evaluator else 0,
    epochs=args.epochs,
    warmup_steps=args.warmup_steps,
    use_amp=True,
    checkpoint_path=args.output_path,
    checkpoint_save_steps=len(train_dataloader),
    checkpoint_save_total_limit=3,
    optimizer_params={"lr": args.lr},
    show_progress_bar=True,
    output_path=args.output_path,
    callback=training_logger,
)

# ============================================================================
# 8. 儲存模型
# ============================================================================

logging.info("=" * 70)
logging.info("訓練完成！儲存模型")
logging.info("=" * 70)

model.save(args.output_path)

logging.info(f"模型已儲存至: {args.output_path}")
logging.info(f"訓練記錄已儲存至: {training_logger.log_file}")
