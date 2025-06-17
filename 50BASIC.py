import json
import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🔹 Parquet 파일 로드
df = pd.read_parquet("eLife_reordered.parquet")
print(f"총 article 개수: {len(df)}")
def extract_clean_summary(full_output):
    """'Now, write the summary:' 이후 ~ 'summary end' 또는 줄바꿈 전까지 텍스트 추출"""
    start_match = re.search(r"Now, write the summary:\s*", full_output)
    if not start_match:
        return ""
    start_idx = start_match.end()
    end_match = re.search(r"summary end", full_output[start_idx:], re.IGNORECASE)
    end_idx = start_idx + end_match.start() if end_match else None
    summary_body = full_output[start_idx:end_idx].strip()
    return summary_body

# 🔹 결과 저장용 리스트
results = []

# 🔁 1~5번 article 반복 처리
for idx in range(50):
    article = df.loc[idx, "article"]

    # 모델 로드 (매번 새로)
    model_path = "/data/a5252545/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # 프롬프트 구성
    prompt = f"""
        "You are a biomedical writing assistant. Generate a high-quality lay summary of the given biomedical research article. "
        "Your summary should be understandable to non-expert readers.\n"
        ### Article:{article}
        Now, write the summary:
        """.strip()


    # 모델 실행
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1200,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    cleaned_summary = extract_clean_summary(full_output)

    results.append({
        "index": idx,
        "summary": cleaned_summary
    })

    print(f"[{idx+1}/50] 요약 완료")

    # 메모리 정리
    del model
    del tokenizer
    torch.cuda.empty_cache()

# 🔹 CSV 저장
summary_df = pd.DataFrame(results)
summary_df.to_csv("50BasicSummary.csv", index=False, encoding="utf-8")
print("✅ 모든 요약 완료 및 저장되었습니다.")
