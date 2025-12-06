---

# Movie_review_sentiment_analysis

## 🧠 Transformer 기반 한국어 감성 분석 프로젝트

---

**한국어 영화 리뷰(NSMC)를 기반으로 KoBERT / KoELECTRA 모델을 파인튜닝하여 문장 단위 감성을 분류하는 프로젝트입니다.**
<br>(이진 분류: 긍정 / 부정)

---

📌 프로젝트 개요

**프로젝트명:** Movie Review Sentiment Analysis
<br>**사용 모델:**
<br>  - 🟦 KoBERT (SKTBrain)
<br>  - 🟩 KoELECTRA (monologg)
<br>**데이터셋:** NSMC (Naver Sentiment Movie Corpus)
<br>**문제 유형:** 문장 단위 감성 분석 (Binary Classification)
<br>**프레임워크:** PyTorch, Hugging Face Transformers
<br>**개발 환경:** Google Colab (무료 GPU + CPU)

---

**📂 프로젝트 구조**

```
Movie_review_sentiment_analysis/
  ├─ kobert_nsmc_exp2/              # Fine-tuned KoBERT 모델 (LFS)
  ├─ koelectra_nsmc_exp1/           # Fine-tuned KoELECTRA 모델 (LFS)
  ├─ nsmc_train_kobert.ipynb        # KoBERT 학습 노트북
  ├─ nsmc_train_koelectra.ipynb     # KoELECTRA 학습 노트북
  ├─ inference_cpu.py               # CPU 환경 추론 스크립트
  └─ README.md
```

---

**📘 데이터셋(NSMC)**

- 네이버 영화 리뷰 200,000건
- Train: 150,000 / Test: 50,000
- 긍정(1), 부정(0) 레이블 구성
- 전처리: 결측/공백 제거, 128 토큰으로 패딩

---

**🧩 모델 설명**

**🔵 KoBERT (SKTBrain)**
- 한국어 형태에 최적화된 BERT 모델
- 토크나이저: SentencePiece
- 베이스라인 모델로 사용

**🟢 KoELECTRA (monologg)**
- ELECTRA Discriminator 기반
- BERT 대비 훈련 효율 높고 성능 우수
- 실험 결과 본 프로젝트에서 가장 높은 성능 기록

---

**⚙️ 학습 환경**

- Google Colab Free GPU(T4)
- 학습 설정:

| 항목         | 값                             |
| ---------- | ----------------------------- |
| Max Length | 128                           |
| Batch Size | 32                            |
| Optimizer  | AdamW                         |
| LR         | 5e-5(KoBERT), 3e-5(KoELECTRA) |
| Epochs     | 3                             |
| Loss       | Cross Entropy                 |

---

**📊 실험 결과**

**KoBERT vs KoELECTRA 성능 비교**
| Exp | Backbone                                               | Epoch | LR   | Valid Acc  | Valid F1   |
| --- | ------------------------------------------------------ | ----- | ---- | ---------- | ---------- |
| 1   | KoBERT (`monologg/kobert`)                             | 3     | 5e-5 | **0.8862** | **0.8879** |
| 2   | KoELECTRA (`monologg/koelectra-base-v3-discriminator`) | 3     | 3e-5 | **0.9112** | **0.9125** |

👉 **KoELECTRA가 약 +2.5%p 더 높은 정확도**를 보이며 가장 우수한 성능을 기록함.<br>
👉 Epoch 3에서 KoELECTRA는 약간의 과적합 조짐이 있으나, F1-score는 최고치를 기록.

---

**🤖 추론(Inference) 예시 (CPU 환경 가능)**

**▶️ 예시 코드**
```
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cpu")

# KoELECTRA 모델 로드
model_dir = "./koelectra_nsmc_exp1"
tokenizer = AutoTokenizer.from_pretrained(
    "monologg/koelectra-base-v3-discriminator",
    use_fast=True,
    trust_remote_code=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    local_files_only=True,
)
model.to(device)
model.eval()

MAX_LEN = 128
id2label = {0: "부정", 1: "긍정"}

def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred = int(torch.argmax(probs))
        return id2label[pred], float(probs[0][pred])

# 테스트 문장
sentences = [
    "진짜 너무 재밌는 영화였다. 또 보고 싶다.",
    "스토리도 별로고 연기도 어색해서 시간 아까웠다.",
    "그냥저냥 볼 만했지만 다시 볼 정도는 아니다.",
    "돈이 전혀 아깝지 않았고, 시간 가는 줄 몰랐다.",
    "뭘 말하고 싶은지도 모르겠고 전체적으로 난해함",
]

for s in sentences:
    label, prob = predict_sentiment(s)
    print(f"{s} → {label} ({prob:.4f})")
```

---

**📈 실제 예측 결과**
```
"진짜 너무 재밌는 영화였다. 또 보고 싶다." → 긍정 (0.9910)
"스토리도 별로고 연기도 어색해서 시간 아까웠다." → 부정 (0.9988)
"그냥저냥 볼 만했지만 다시 볼 정도는 아니다." → 부정 (0.9978)
"돈이 전혀 아깝지 않았고, 시간 가는 줄 몰랐다." → 긍정 (0.9971)
"전체적으로 난해하고 뭘 말하고 싶은지도 모르겠다." → 부정 (0.9988)
```

---

**🔮 향후 개선 방안**

- 다중 모델 앙상블
- 사전학습 RoBERTa 기반 모델 비교
- 리뷰 길이가 긴 문장을 위한 Longformer 계열 모델 적용
- 감성 강도(슬픔/기쁨/분노 등) 다중 레이블 확장

---

**📜 라이선스**

- NSMC 데이터는 Naver & e9t의 라이선스를 따릅니다.
- KoBERT는 SKTBrain 라이선스를 따릅니다.
- KoELECTRA는 monologg의 공개 모델을 사용합니다.

---

**🔗 Reference**

[NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)<br>
[KoBERT (SKT)](https://github.com/SKTBrain/KoBERT)<br>
[KoELECTRA (monologg)](https://github.com/monologg/KoELECTRA)
