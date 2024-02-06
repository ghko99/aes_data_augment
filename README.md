
## 📝  데이터 증강을 이용한 KoBERT 기반 에세이 자동 평가 성능 향상

ICCMST 2022에 올라온 논문 [Data Augmentation for Automated Essay Scoring using Transformer Models](https://arxiv.org/abs/2210.12809)에서는 Transformers 기반 Automated Essay Scoring 모델 성능 향상을 위한 효과적인 데이터 증강법을 제안함. 해당 증강법을 AI-HUB에 올라온 한국어 에세이 글 평가 데이터에 적용해 기존의 KoBERT 기반 AES 모델의 성능을 향상 시켜보았음.


## 제안 하는 방식
기존의 Automated Essay Scoring 모델들의 단점은 모두 주제별로 다 다르다는 점임. 해당 논문에서는 데이터 증강법을 통해 데이터가 풍부한 주제뿐만 아니라, 데이터가 부족한 주제에서도 좋은 성과를 낼 수 있는 Automated Essay Scoring 모델의 시스템을 만든다는 것임.

다음과 같이 에세이의 Topic 정보를 에세이의 특정 라인에 삽입해 증강하면, Automated Essay Scoring 모델이 topic-specific 하게 학습하기 때문에 적은 양의 데이터로 낮은 성능을 보이는 주제의 에세이의 성능을 끌어올려 전체적인 모델의 성능을 올릴 수 있다고 함.

해당 방식을 다음 그림과 같이 AI-HUB 에세이 글 평가 데이터에 적용해본뒤 KoBERT-GRU 기반 AES 모델에 학습시켜봤음.
[monologg/kobert](https://github.com/monologg/KoBERT-Transformers)를 통해 에세이의 문장별 임베딩을 추출한 뒤 GRU에서 임베딩 값을 받아 에세이를 scoring. 총 11개의 평가 기준에 따른 점수를 출력

 ![default](image/model.png)

## 코드 구성 (에세이 Embedding 추출)
```bash
python3 aes_embedding.py
```
dataset.csv에 저장된 에세이 원문의 문장별 임베딩 값을 csv 파일에 저장.

## 코드 구성 (Train)
```bash
python3 aes_train.py
```
csv 파일에 저장된 임베딩 값을 tensorflow기반 gru 모델에 입력해 에세이 점수를 학습하고 kappa score, pearson 상관계수를 측정함

## 모델 성능 측정 결과
```
Kappa Score 1 : 0.5587065386538349
Pearson Correlation Coefficient 1 : 0.6135622669936321
```


## 데이터 구조
![data](image/data.png)

## Label (rubric) 구성
11가지 평가 기준에 따른 0~3점 사이의 정수 점수로 이루어짐
각 평가기준은 논술형, 수필형에 따라 다름

* 논술형 루브릭 구성
![non](image/non.png)
* 수필형 루브릭 구성
![su](image/su.png)

각 루브릭 별 가중치에 따라 총점을 계산해 kappa score, pearson correlation 측정