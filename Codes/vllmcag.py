import time
import pandas as pd
import re
import uuid
from openai import OpenAI
# srun -p debug_ugrad --gres=gpu:1 -w aurora-g1 --cpus-per-gpu=8 --mem-per-gpu=29G --pty $SHELL

# tmux new -s vllm_server

# tmux split-window -h

# python3 -m vllm.entrypoints.openai.api_server \
# --model /data/a5252545/model \
# --tokenizer /data/a5252545/model \
# --dtype bfloat16 \
# --max-model-len 8192 \
# --enable-prefix-caching \
# --served-model-name summary-test

# ▶︎ 1. vLLM 서버 연결
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# ▶︎ 2. 데이터 로딩 (원하면 .head(n)으로 줄이기)
df = pd.read_parquet("/data/a5252545/capstone/eLife_reordered.parquet", engine="pyarrow").head(50)

# ▶︎ 3. 프롬프트 생성
def make_system(article):
    return {
        "role": "system",
        "content": (
            "You are a biomedical writing assistant. Generate a high-quality lay summary of the given biomedical research article. "
            "Your summary should be understandable to non-expert readers, such as high school students or the general public, without losing essential scientific content.\n\n"
            "Write as a single coherent paragraph.\n"
            "Instructions:\n"
            "- Start the summary directly without repeating the title.\n"
            "- Do NOT include section headers, key findings, or bullet points.\n"
            "- End the summary with the phrase: summary end\n\n"
            f"### Article:\n{article}"
        )
    }

def make_user():
    return {
        "role": "user",
        "content": "Now, write the summary:"
    }

# ▶︎ 4. 요약 추출
def extract_clean_summary(full_output):
    end_match = re.search(r"summary end", full_output, re.IGNORECASE)
    end_idx = end_match.start() if end_match else None
    return full_output[:end_idx].strip()

# ▶︎ 5. 실험 루프
results = []
for idx, row in df.iterrows():
    session_id = str(uuid.uuid4())  # 동일 article에 대해 동일한 세션 유지

    sys_msg = make_system(row["article"])
    user_msg = make_user()
    messages = [sys_msg, user_msg]

    for trial in range(5):
        t0 = time.time()
        response = client.chat.completions.create(
            model="summary-test",
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            extra_headers={"X-Session-ID": session_id}
        )
        t1 = time.time()

        results.append({
            "article_index": idx,
            "trial": trial + 1,
            "elapsed_time_sec": round(t1 - t0, 3),
            "summary": extract_clean_summary(response.choices[0].message.content)
        })

        print(f"[{idx+1}/50][Trial {trial+1}/5] ⏱️ {t1 - t0:.2f}s")

# ▶︎ 6. 저장
pd.DataFrame(results).to_csv("summary_true_cag_sessioned_5x.csv", index=False)
print("✅ summary_true_cag_sessioned_5x.csv 저장 완료.")
