import os
os.environ["HF_HOME"] = "/data/a5252545/huggingface_cache"

import pandas as pd
import nltk
import textstat
import bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu


# 🔹 데이터 로딩
original_df = pd.read_csv("/data/a5252545/capstone/original_summaries.csv")
cag_df = pd.read_csv("/data/a5252545/capstone/summary_50_kvcache.csv")  # KV-cache 결과 (50개)

# 🔹 ROUGE 계산 함수
def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores_list = [scorer.score(r, p) for p, r in zip(preds, refs)]
    return {
        "ROUGE-1": sum(s['rouge1'].fmeasure for s in scores_list) / len(scores_list),
        "ROUGE-2": sum(s['rouge2'].fmeasure for s in scores_list) / len(scores_list),
        "ROUGE-L": sum(s['rougeL'].fmeasure for s in scores_list) / len(scores_list)
    }

# 🔹 BLEU 계산 함수
def compute_bleu(preds, refs):
    refs_list = [[ref.split()] for ref in refs]
    preds_list = [pred.split() for pred in preds]
    return corpus_bleu(refs_list, preds_list)

# 🔹 종합 metric 계산 함수
def compute_metrics(preds, refs, gen_times=None):
    rouge = compute_rouge(preds, refs)
    bleu = compute_bleu(preds, refs)
    P, R, F1 = bert_score.score(preds, refs, lang="en", model_type="distilbert-base-uncased")
    fkgl = sum(textstat.flesch_kincaid_grade(p) for p in preds) / len(preds)
    
    result = {
        "ROUGE-1": rouge["ROUGE-1"],
        "ROUGE-2": rouge["ROUGE-2"],
        "ROUGE-L": rouge["ROUGE-L"],
        "BLEU": bleu,
        "BERTScore (F1)": F1.mean().item(),
        "Readability (FKGL)": fkgl
    }

    if gen_times is not None:
        result["Gen Time (sec)"] = sum(gen_times) / len(gen_times)
    
    return result

# 🔹 단일 평가 실행
subset = cag_df.sort_values("index")
preds = subset["summary"].tolist()
refs = original_df[original_df["index"].isin(subset["index"])].sort_values("index")["summary"].tolist()
gen_times = subset["gen_time_sec"].tolist()

# 🔹 Metric 계산 및 저장
final_metrics = compute_metrics(preds, refs, gen_times)
metrics_df = pd.DataFrame([final_metrics])
metrics_df.to_csv("summary_metrics_kvcache_single.csv", index=False)
print(metrics_df)
