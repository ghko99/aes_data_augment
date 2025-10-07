
## 📝  데이터 증강을 이용한 KoBERT 기반 에세이 자동 평가 성능 향상

* ICCMST 2022에 올라온 논문 [Data Augmentation for Automated Essay Scoring using Transformer Models](https://arxiv.org/abs/2210.12809)에서는 Transformers 기반 Automated Essay Scoring 모델 성능 향상을 위한 효과적인 데이터 증강법을 제안함. 해당 증강법을 AI-HUB에 올라온 한국어 에세이 글 평가 데이터에 적용해 기존의 KoBERT 기반 AES 모델의 성능을 향상 시켜보았음.


## 제안 하는 방식
* 기존의 Automated Essay Scoring 모델들의 단점은 모델이 주제별로 다르다는 점임. 해당 논문에서는 데이터 증강법을 통해 데이터가 풍부한 주제뿐만 아니라, 데이터가 부족한 주제에서도 좋은 성과를 낼 수 있는 Automated Essay Scoring 모델의 시스템을 만든다는 것임.

![스크린샷 2024-02-06 200341](https://github.com/ghko99/aes_data_augment/assets/115913818/f3f7e876-0a64-436e-8853-15e84baefa5f)

* 다음과 같이 에세이의 Topic 정보를 에세이의 특정 라인에 삽입해 증강하면, Automated Essay Scoring 모델이 topic-specific 하게 학습하기 때문에 적은 양의 데이터로 낮은 성능을 보이는 주제의 에세이의 성능을 끌어올려 전체적인 모델의 성능을 올릴 수 있다고 함.

![스크린샷 2024-02-06 200444](https://github.com/ghko99/aes_data_augment/assets/115913818/a6e7edd5-53c0-4422-94bc-6844c9b41026)

* 해당 방식을 다음 그림과 같이 AI-HUB 에세이 글 평가 데이터에 적용해본뒤 KoBERT-GRU 기반 AES 모델에 학습시켜봤음.
ICCMST 논문에 따르면 Topic을 10문장마다 삽입하는 것이 좋은 성능을 보였다고함. 하지만 ASAP데이터셋과 달리 에세이 글 평가 데이터는 비교적 적은 문장으로 이루어진 에세이 데이터셋이기 때문에 10문장이 아닌 1문장마다 Topic을 삽입함.

## 필요 라이브러리 (Dependencies)

본 프로젝트를 실행하기 위해 필요한 파이썬 라이브러리는 다음과 같습니다.

```
torch
pandas
tensorflow
transformers
tqdm
scikit-learn
sentencepiece
```

`requirements.txt` 파일을 생성하여 관리하거나, 아래 명령어를 통해 한 번에 설치할 수 있습니다.

```bash
pip install torch pandas tensorflow transformers tqdm scikit-learn sentencepiece
```

## 성능 측정 방법
* 코드 구성-> aes_embedding.py에서 에세이의 임베딩 벡터 추출후, csv파일에 저장. aes_train.py에서 csv파일 read후 gru 모델에 학습
![image](https://github.com/ghko99/aes_data_augment/assets/115913818/24768dc0-4c6a-4d31-988e-6c24dc40dfc5)
* baseline 모델 성능 측정
  ```bash
  python3 aes_embedding.py --is_topic=False
  ```
  ```bash
  python3 aes_train.py --is_topic=False
  ```
* data augmentation 적용 모델 성능 측정
  ```bash
  python3 aes_embedding.py --is_topic=True
  ```
  ```bash
  python3 aes_train.py --is_topic=True
  ```

## 모델 성능 측정 결과
* baseline
  ```
  Kappa Score 1 : 0.5587065386538349
  Pearson Correlation Coefficient 1 : 0.6135622669936321
  ```
* data augmentation 적용 모델 성능
  ```
  Kappa Score 1 : 0.5917502753070988
  Pearson Correlation Coefficient 1 : 0.6250704841699577
  ```

## Reference
* [에세이 글 평가 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=545)
* [monologg/kobert](https://github.com/monologg/KoBERT-Transformers)
* [Data Augmentation for Automated Essay Scoring using Transformer Models](https://arxiv.org/abs/2210.12809)
* [데이터 증강을 이용한 KoBERT 기반 에세이 자동 평가 성능 향상](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11488531)
