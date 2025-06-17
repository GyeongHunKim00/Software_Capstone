import bz2
import re
import json
from tqdm import tqdm

def build_redirect_dict(xml_dump_path, save_path="redirect_dict.json"):
    redirect_pattern = re.compile(r'<redirect title="([^"]+)"')
    title_pattern = re.compile(r'<title>(.*?)</title>')

    redirect_map = {}
    with bz2.open(xml_dump_path, "rt", encoding="utf-8") as f:
        title = None
        for line in tqdm(f, desc="🔄 Parsing XML dump for redirects"):
            if "<page>" in line:
                title = None  # reset title
            elif "<title>" in line:
                match = title_pattern.search(line)
                if match:
                    title = match.group(1)
            elif "<redirect title=" in line and title:
                match = redirect_pattern.search(line)
                if match:
                    redirect_target = match.group(1)
                    redirect_map[title] = redirect_target

    # 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(redirect_map, f, ensure_ascii=False, indent=2)

    print(f"✅ Redirect 매핑 {len(redirect_map):,}개 저장 완료 → {save_path}")
    return redirect_map

# 실행
xml_path = "/data/a5252545/capstone/wikipedia/enwiki-20250401-pages-articles-multistream.xml.bz2"
build_redirect_dict(xml_path)
