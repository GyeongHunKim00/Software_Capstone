import json
import torch
import pandas as pd
import re
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
# ✅ Metric 초기화

# ✅ CUI 정의 로딩
print("\n🔄 Loading CUI definition table...")
cui_def_df = pd.read_csv("/data/a5252545/capstone/real_cui_wiki_def_enhanced.csv")

# ✅ spaCy NER 모델 로드 (AVX 없는 환경이므로 Entity Linking 제외)
print("🔄 Loading spaCy model...")
nlp = spacy.load("en_ner_bionlp13cg_md")

# 2. MRSTY.RRF → CUI to TUI 로딩 추가
print("🔄 Loading MRSTY (semantic type)...")
mrsty_df = pd.read_csv("/data/a5252545/capstone/UMLS/MRSTY.RRF", sep='|', header=None, dtype=str)
mrsty_df = mrsty_df.iloc[:, :4]
mrsty_df.columns = ["CUI", "TUI", "STY", "ATUI"]
cui_to_tui = mrsty_df.groupby("CUI")["TUI"].apply(set).to_dict()
TUI_WHITELIST = {
    "T116",  # Amino Acid, Peptide, or Protein
    "T121",  # Pharmacologic Substance
    "T047",  # Disease or Syndrome
    "T123",  # Biologically Active Substance
    "T022",  # Body System
    "T025",  # Cell
    "T026",  # Cell Component
    "T028",  # Gene or Genome
    "T020",  # Biological Function
    "T087",  # Molecular Function
    "T129",  # Immunologic Factor
    "T109",  # Organic Chemical → edta, tryptophan 등 포함 가능
    "T114",  # Nucleic Acid → DNA
}
# ✅ 생물학 엔티티를 target label 필터링하고 빈도수 기준 상위 15개 term 선정 함수
def get_umls_terms_by_frequency(article_text, top_n=15):
    doc = nlp(article_text)
    freq = {}
    for ent in doc.ents:
        normalized = ent.lemma_.strip().lower()
        freq[normalized] = freq.get(normalized, 0) + 1
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in sorted_terms[:top_n]]

def match_term_to_definition(term: str):
    matches = cui_def_df[cui_def_df["STR"].str.lower().str.contains(rf"\b{re.escape(term)}\b", na=False)]
    if matches.empty:
        matches = cui_def_df[cui_def_df["STR"].str.lower().str.startswith(term)]
    if matches.empty:
        return None, None
    row = matches.iloc[0]
    return row["CUI"], row["definition"]



def is_cui_meaningful(cui: str) -> bool:
    tui_set = cui_to_tui.get(cui, set())
    return bool(tui_set & TUI_WHITELIST)

# ✅ 정의 테이블에서 term 기준 매핑하여 KG 구축 함수
def get_kg_context(article_text, top_n=15, debug=False):
    terms = get_umls_terms_by_frequency(article_text, top_n=top_n)
    kg_entries, seen_cuis = [], set()

    if debug:
        print(f"\n📌 Extracted terms: {terms}")

    for term in terms:
        cui, definition = match_term_to_definition(term)
        if not cui or cui in seen_cuis:
            continue
        if not is_cui_meaningful(cui):
            if debug:
                print(f"⛔ {term} (CUI: {cui}) skipped due to TUI: {cui_to_tui.get(cui, set())}")
            continue
        kg_entries.append(f"{term.upper()}: {definition}")
        seen_cuis.add(cui)

    return "\n".join(kg_entries)
# ✅ LLaMA 출력에서 요약 부분 추출 함수
def extract_clean_summary(full_output):
    start_match = re.search(r"Now, write the summary:\s*", full_output)
    if not start_match:
        return ""
    start_idx = start_match.end()
    end_match = re.search(r"summary end", full_output[start_idx:], re.IGNORECASE)
    end_idx = start_idx + end_match.start() if end_match else None
    summary_body = full_output[start_idx:end_idx].strip()
    return summary_body

# ✅ eLife 논문 데이터 로드
df = pd.read_parquet("eLife_reordered.parquet")
print(f"총 article 개수: {len(df)}")

results = []
for idx in range(50):
    article = df.loc[idx, "article"]
    kg_context = get_kg_context(article)



    prompt = f"""
You are a biomedical writing assistant. Generate a high-quality lay summary of the given biomedical research article. Your summary should be understandable to non-expert readers, such as high school students or the general public, without losing essential scientific content.

Follow these instructions:

🧠 Content & Structure
Include all core information: What was studied, why it matters, how it was done, and what was discovered.

Maintain logical flow: Present information in this order – problem → method → results → significance.

Write as a single coherent paragraph. Do not use headings, bullets, or multiple paragraphs.
Incorporate background knowledge briefly when helpful (e.g., the function of a protein). Use the provided glossary (`{kg_context}`) only when terms appear in the summary. If used, naturally explain the term using the glossary definition. Do not insert glossary terms that are irrelevant.

At the end of the summary, write "summary end".

Do not use section headers (e.g., “Key biological finding”, "## Step N", “Mechanisms”, “Experimental Observations”) or line breaks.

💬 Language & Clarity
Avoid repetition, metaphors, or storytelling: Do not use analogies or figurative language. Stick to clear, factual explanation.

Use simple, direct language: Avoid technical jargon. If a technical term must be used (e.g., "apoptosis", "rhizobia"), mention it only once and explain it clearly in plain English.

Prefer short and active sentences: Break up long, complex sentences into two. Use subject-verb-object order.

📏 Length & Style
Keep the summary around 250–450 words.
Target FKGL score: 10–12.
Begin the summary directly with the main content. Do not use introductory phrases such as “This study shows”, “Researchers found that”, or “Scientists studying”. Avoid narrative or conversational framing. Write in a neutral, impersonal tone.

### Article:
{article}
Now, write the summary:
""".strip()

    tokenizer = AutoTokenizer.from_pretrained("/data/a5252545/model")
    model = AutoModelForCausalLM.from_pretrained("/data/a5252545/model", torch_dtype=torch.bfloat16, device_map="auto")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1200, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.2)
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        summary = extract_clean_summary(full_output)

    results.append({"index": idx, "summary": summary})
    print(f"[{idx+1}/50] 요약 완료 ✅")
    print(kg_context)
    del model, tokenizer
    torch.cuda.empty_cache()

pd.DataFrame(results).to_csv("50RagSummary.csv", index=False, encoding="utf-8")
print("✅ 모든 요약 완료 및 저장되었습니다.")
