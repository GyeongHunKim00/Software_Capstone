import pandas as pd
import nltk
import textstat
import bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu

nltk.download('punkt')

original_path = "/data/a5252545/capstone/original_summaries.csv"
kg_path = "/data/a5252545/capstone//50RagSummary.csv"
basic_path = "/data/a5252545/capstone//50BasicSummary.csv"

original_df = pd.read_csv(original_path)
kg_df = pd.read_csv(kg_path)
basic_df = pd.read_csv(basic_path)

references = original_df["summary"].tolist()
kg_preds = kg_df["summary"].tolist()
basic_preds = basic_df["summary"].tolist()

def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_f1, rouge2_f1, rougeL_f1 = [], [], []
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        rouge1_f1.append(scores['rouge1'].fmeasure)
        rouge2_f1.append(scores['rouge2'].fmeasure)
        rougeL_f1.append(scores['rougeL'].fmeasure)
    
    return {
        "ROUGE-1": sum(rouge1_f1) / len(rouge1_f1),
        "ROUGE-2": sum(rouge2_f1) / len(rouge2_f1),
        "ROUGE-L": sum(rougeL_f1) / len(rougeL_f1)
    }

def compute_bleu(preds, refs):
    refs_list = [[ref.split()] for ref in refs]  
    preds_list = [pred.split() for pred in preds]  
    return corpus_bleu(refs_list, preds_list)

def compute_metrics(preds, refs):
    rouge_result = compute_rouge(preds, refs)
    
    bleu_result = compute_bleu(preds, refs)
    
    P, R, F1 = bert_score.score(preds, refs, lang="en")
    
    readability = sum([textstat.flesch_kincaid_grade(p) for p in preds]) / len(preds)

    return {
        "ROUGE-1": rouge_result["ROUGE-1"],  # ROUGE-1 F1 score
        "ROUGE-2": rouge_result["ROUGE-2"],  # ROUGE-2 F1 score
        "ROUGE-L": rouge_result["ROUGE-L"],  # ROUGE-L F1 score
        "BLEU": bleu_result,  # BLEU score
        "BERTScore (F1)": F1.mean().item(),  # BERTScore F1 score
        "Readability (FKGL)": readability  # Average Flesch-Kincaid Grade Level
    }

metrics_kg = compute_metrics(kg_preds, references)
metrics_basic = compute_metrics(basic_preds, references)

print("📏 Reference 평균 길이:", sum(len(r.split()) for r in references) / len(references))
print("📏 KG 평균 길이:", sum(len(p.split()) for p in kg_preds) / len(kg_preds))
print("📏 Basic 평균 길이:", sum(len(p.split()) for p in basic_preds) / len(basic_preds))

metrics_df = pd.DataFrame([metrics_kg, metrics_basic], index=["RAG", "Basic"])
metrics_df.to_csv("50summary_evaluation_metrics.csv")
print("✅ 평가 완료: summary_evaluation_metrics.csv 파일로 저장되었습니다.")
print(metrics_df)
