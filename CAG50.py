import torch
import time
import os
import re
import uuid
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# ✅ 모델 로딩
model_name = "/data/a5252545/model"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_buffers=False
)

# ✅ KV 캐시 생성 함수
def get_kv_cache(text: str) -> DynamicCache:
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    past_key_values = DynamicCache()
    with torch.no_grad():
        _ = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    return past_key_values

# ✅ 요약 생성 함수
def generate_summary(prompt: str, kv_cache: DynamicCache, max_new_tokens: int = 300) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated_ids = input_ids
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_token, past_key_values=kv_cache, use_cache=True)
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1).unsqueeze(0).to(model.device)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

# ✅ 논문 로딩
df = pd.read_parquet("/data/a5252545/capstone/eLife_reordered.parquet", engine="pyarrow").head(50)
results = []

# ✅ 50개 문서 처리
for idx, row in df.iterrows():
    article_text = row["article"]
    article_id = row.get("id", str(idx))  # 혹시 id 컬럼이 있으면 사용

    # ▶︎ 1. system 프롬프트 구성
    system_prompt = (
        "<s>[INST] <<SYS>>\n"
        "You are a biomedical writing assistant. Generate a high-quality lay summary of the given biomedical research article. "
        "Your summary should be understandable to non-expert readers.\n"
        "<</SYS>>\n\n"
        f"### Article:\n{article_text}\nNow, write the summary:\n"
        "[/INST]"
    )

    # ▶︎ 2. KV 캐시 생성
    start = time.time()
    kv_cache = get_kv_cache(system_prompt)
    cache_time = time.time() - start

    # ▶︎ 3. 디코딩 전용 질의
    query_prompt = "Now, write the summary:\n"
    start = time.time()
    try:
        summary = generate_summary(query_prompt, kv_cache)
    except Exception as e:
        summary = f"ERROR: {str(e)}"
    gen_time = time.time() - start

    # ▶︎ 4. 결과 저장
    results.append({
        "index": idx,
        "article_id": article_id,
        "cache_time_sec": round(cache_time, 3),
        "gen_time_sec": round(gen_time, 3),
        "summary": summary.strip()
    })

    print(f"[{idx+1}/50] ✅ {round(cache_time, 3)}s (cache) / {round(gen_time, 3)}s (gen)")

    # ▶︎ 5. GPU 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ✅ 최종 저장
out_path = "/data/a5252545/capstone/summary_50_kvcache.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\n✅ 저장 완료: {out_path}")
