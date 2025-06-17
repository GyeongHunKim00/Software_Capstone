import pandas as pd
import re
import bz2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ✅ 경로 설정
MRCONSO_PATH = "/data/a5252545/capstone/UMLS/MRCONSO.RRF"
MRDEF_PATH = "/data/a5252545/capstone/UMLS/MRDEF.RRF"
REDIRECT_DICT_PATH = "/data/a5252545/capstone/redirect_dict.json"
WIKI_DUMP_PATH = "/data/a5252545/capstone/wikipedia/enwiki-20250401-pages-articles-multistream.xml.bz2"
OUTPUT_PATH = "real_cui_wiki_def_enhanced.csv"

# ✅ 1. Wikipedia title normalization 함수
def normalize_title(s):
    if not isinstance(s, str):
        return None
    s = re.sub(r'\(.*?\)', '', s)        # 괄호 제거
    s = re.sub(r'\s+', ' ', s).strip()   # 공백 정리
    s = re.sub(r'[^\w\s\-]', '', s)    # 특수문자 제거
    s = s.capitalize()                   # 첫글자 대문자
    return s

# ✅ 2. STR 목록 불러오기
print("🔹 Loading MRCONSO...")
conso = pd.read_csv(MRCONSO_PATH, sep='|', header=None, dtype=str, usecols=[0, 11, 14])
conso.columns = ['CUI', 'SAB', 'STR']
conso = conso.dropna(subset=['STR'])
conso = conso[conso['SAB'].isin(['MSH', 'SNOMEDCT_US', 'MEDLINE'])].drop_duplicates(subset=['CUI', 'STR'])
conso['norm_title'] = conso['STR'].apply(normalize_title)

# ✅ 3. Redirect 사전 불러오기
import json
print("🔹 Loading Redirect dict...")
with open(REDIRECT_DICT_PATH, 'r') as f:
    redirect_dict = json.load(f)

# ✅ 4. UMLS 정의 로드
def_df = pd.read_csv(MRDEF_PATH, sep='|', header=None, usecols=[0, 5], names=['CUI', 'DEF'], dtype=str)
mrdef_map = def_df.groupby('CUI')['DEF'].first().to_dict()

# ✅ 5. Wikipedia 정의 매핑 함수
def parse_wikipedia_dump(dump_path, titles_set):
    wiki_defs = {}
    print("🔹 Parsing Wikipedia XML dump...")
    with bz2.open(dump_path, 'rt', encoding='utf-8') as f:
        context = ET.iterparse(f, events=('end',))
        for event, elem in tqdm(context, desc="Parsing", unit=" pages"):
            if elem.tag.endswith('page'):
                title = elem.findtext('./title')
                text = elem.findtext('./revision/text')
                if title in titles_set and text:
                    first_sentence = extract_first_sentence(text)
                    if first_sentence:
                        wiki_defs[title] = first_sentence
                elem.clear()
    return wiki_defs

def extract_first_sentence(text):
    candidates = re.split(r'\.(\s|\n)', text)
    for sent in candidates:
        sent = sent.strip()
        if len(sent) > 20:
            return sent + '.'
    return None

# ✅ 6. 전체 타이틀 셋 만들기 (정규화 + redirect 포함)
norm_titles = set(conso['norm_title'].dropna())
redirected_titles = {redirect_dict.get(t, t) for t in norm_titles}
titles_to_match = norm_titles.union(redirected_titles)

# ✅ 7. Wikipedia 텍스트 정의 매핑 수행
wiki_def_map = parse_wikipedia_dump(WIKI_DUMP_PATH, titles_to_match)

# ✅ 8. term에 대해 위키 정의 or fallback 정의 삽입
def get_best_definition(cui, raw_str, norm_title):
    # 우선순위 1: redirect → wiki
    redirected = redirect_dict.get(norm_title, norm_title)
    if redirected in wiki_def_map:
        return wiki_def_map[redirected], 'wiki'
    # 우선순위 2: wiki 직매핑
    if norm_title in wiki_def_map:
        return wiki_def_map[norm_title], 'wiki'
    # 우선순위 3: UMLS 정의
    if cui in mrdef_map:
        return mrdef_map[cui], 'umls'
    return None, 'none'

# ✅ 9. 최종 정의 테이블 생성
records = []
for _, row in tqdm(conso.iterrows(), total=len(conso)):
    cui, s, norm = row['CUI'], row['STR'], row['norm_title']
    definition, source = get_best_definition(cui, s, norm)
    if definition:
        records.append({
            'CUI': cui,
            'STR': s,
            'definition': definition,
            'source': source
        })

# ✅ 10. 저장
df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
print(f"✅ 정의 매핑 완료 → 총 {len(df_out)}개 term 저장 → {OUTPUT_PATH}")