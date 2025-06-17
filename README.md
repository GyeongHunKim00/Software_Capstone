# 바이오메디컬 문헌의 대중 친화적 요약 성능 향상을 위한 용어 정의 기반 Prompt 전략 방법

> 용어집(glossary)을 활용한 경량 RAG 프레임워크를 통해, 일반 독자를 위한 생의학 요약 품질을 향상시키는 프로젝트입니다.

## 개요

본 프로젝트는 복잡한 생명과학 및 의학 논문을 비전문 독자도 이해할 수 있도록 요약하는 것을 목표로 하며, **Wikipedia** 및 **UMLS**로부터 수집한 용어-정의 쌍을 요약 프롬프트에 사전 삽입하는 간단한 **glossary 기반 RAG 프레임워크**를 제안합니다.

별도의 검색 시스템이나 지식 그래프 없이도, 외부 정의 기반 지식을 LLM 요약 과정에 자연스럽게 통합할 수 있습니다.

## 실험 설정

- **데이터셋**: [BioLaySumm 2025 – eLife Subset]
- (https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-eLife)
- **엔티티 추출**: `en_ner_bionlp13cg_md` (SciSpacy 사용)
- **용어 정의 추출**:
  - 1순위: Wikipedia (제목 일치 → redirect → 표제어 정규화)
  - 2순위: UMLS MRDEF.RRF 정의 (fallback)
  - TUI 필터링을 통해 질병, 단백질, 유전자 등 주요 생의학 용어만 선정
- **사용 모델**: `LLaMA 3.1 8B Instruct`
- **요약 프롬프트 구성**:
  - glossary 사전 삽입
  - problem → method → result → significance 구조 유지
  - 문단 구분, 소제목, 서사적 표현 금지
  - FKGL 10–12 수준 유지
  - repetition_penalty = 1.2 / max_tokens = 1200

## 요약 성능 비교

| 모델       | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | F1-score ↑ | FKGL ↓ |
|------------|-----------|-----------|-----------|------------|--------|
| Baseline   | 0.3423    | 0.0678    | 0.1548    | 0.8395     | 17.302 |
| 제안 방식 | **0.3747** | **0.0793** | **0.1690** | **0.8438** | **15.736** |

> 간단한 glossary 삽입만으로도 의미 일치도와 문장 가독성 모두에서 개선 효과 확인

## 참고 문헌

1. Luo Z., Xie Q., Ananiadou S., *Readability controllable biomedical document summarization*, Findings of the Association for Computational Linguistics: EMNLP, 2022, pp. 4667–4680.  
2. Hofer M., et al., *Construction of Knowledge Graphs: State and Challenges*, arXiv preprint arXiv:2302.11509, 2023.  
3. Ye Z., et al., *KG-Rank: Enhancing Large Language Models for Medical QA with Knowledge Graphs and Ranking Techniques*, BioNLP Workshop at ACL, 2024, pp. 155–166.  
4. BioLaySumm Team, *BioLaySumm 2025 – eLife Dataset*, Hugging Face, 2025. https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-eLife  
5. Neumann M., et al., *SciSpaCy: Fast and Robust Models for Biomedical NLP*, BioNLP Workshop, 2019, pp. 319–327.  
6. Grattafiori A., et al., *The Llama 3 Herd of Models*, arXiv preprint arXiv:2407.21783, 2024.  
7. Lin C.-Y., *ROUGE: A Package for Automatic Evaluation of Summaries*, ACL Workshop, 2004, pp. 74–81.  
8. Chinchor N., *MUC-4 Evaluation Metrics*, MUC-4 Conference, 1992, pp. 22–29.  
9. Kincaid J.P., et al., *Derivation of New Readability Formulas for Navy Personnel*, Naval Air Station Memphis, 1975.
