# Cost-Effective Active Learning for Melanoma Segmentation

- Reference : [Cost-Effective Active Learning for Melanoma Segmentation][link]

피부 멜라닌종 병변을 식별하는 데 필요한 비용과 시간을 줄이기 위한 방법.

이 방법은 초기에 대표 이미지 집합을 라벨링하고, 모델의 성능을 개선할 수 있는 잠재적인 이미지를 선택하여 반복적으로 추가 훈련시키는 액티브 러닝 접근 방식을 사용. 이러한 방식은 라벨이 지정된 데이터의 효율적인 사용을 가능하게 하며, 기존의 수동적인 학습 방법에 비해 상당한 비용 절감을 이룰 수 있음.

---
# <1> Motivation
1. 문제점
  - 의료 이미지를 해석하기 위해선 전문적인 지식이 필요함 -> 쉽지 않고, cost가 높음
  - 또한, Manual한 검수시 주관적이고 지루, 시간이 많이 소요됨 -> Error 발생하기 쉬움

2. 보완
  - Computer Vision 분야의 많은 알고리즘이 쓰임
  - **Active Learning(AL)** 접근법으로 문제점을 개선할 수 있음
    - 본 논문에서는 선택 기준으로써의 uncertainty of the pixel-wise predictions을 살펴보고자 함

3. 본 논문의 주요 contributions
  - CNN과 CEAL(Cost-Effective Active Learning) 을 사용하여 의료 이미지에 대하여 semantic segmentation을 위한 프레임워크 설계
  - 고유한 네트워크 분포를 분석하기 위하여 *Monte Carlo Dropout*에 근거하여 정보 번역을 발전

  ![image](https://user-images.githubusercontent.com/108987773/218634777-fe488fa5-731e-4227-a837-f0eef69d9e0b.png)

  ```
  Dropout 이란?
  
  Regularization을 적용하는 방법 중에 하나로, 특정 뉴런의 확률 p를 0으로 바꾸는 것
  이전 층에서 계산된 값들이 전달되지 않게 되므로, 모델 수용력의 관점으로 보면 p만큼 각 층의 수용력을 줄임으로써 전체 모델의 수용력을 줄이는 것으로 볼 수 있음(오버피팅될 확률을 피함)

  발생할 수 있는 문제점 - 만약 p가 0.5이고 4개의 노드에서 값이 전달되는 구조였다면, 학습 시에는 확률적으로 평균 2개의 노드에 값이 전달됨. 하지만 테스트 시에는 4개의 노드에서 값이 모두 전달되기 때문에 모델이 학습 때 받던 값의 2배 정도가 입력으로 들어옴.
  이 차이를 맞춰주기 위해 테스트 시에는 드롭 확률을 곱해줌. (p=0.2였다면 테스트 시에는 0.8을 곱해서 전달)
  ```
---
# <2> Related Work
1. Cost-Effective Active Learning(CEAL) algorithm
  ![image](https://user-images.githubusercontent.com/108987773/218636341-e55d08d9-93ca-4682-8867-023522531050.png)
  - 기존의 Active Learning의 접근법과는 반대되는 Cost-Effective methodology가 제시됨
  - Unlabeled Dataset에서의 샘플을 CNN 모델에 입력 -> CNN 결과에 따른 fine-tuning을 위해 **두가지 종류**로 샘플을 나눔
    - **Complementary Sample Selection**
      - Majority Samples with high prediction score : High confidence, CEAL가 자동적으로 pseudo-labels을 붙임
      - Minority Samples with low confidence : Prediction score가 낮음 -> **Oracle에 의한 labeling 필요**

2. CNN's for Image Segmentation: U-Net architecture

  ![image](https://user-images.githubusercontent.com/108987773/218638043-f3f67b41-2c76-4374-99b8-7371e643de7f.png)
  - Medical Domain에선 U-Net(Convolutional + Deconvolutional architecture)이 잘 쓰임
---
# <3> Proposed methodology
## 1. Image uncertainty estimation
### KL Divergence
- Complementary Sample Selection을 위한 active learning은 unlabeled data의 고유한 분포를 알아야함
  - Posterior Distribution(=Parameter Distribution)을 임의로 가정 후 근사치함
  - 두 확률분포(임의로 가정한 분포와 근사치한 분포)의 차이를 최소화하여 network의 가중치 q(w)를 구할 수 있음
  ```
  고유의 분포를 어떻게 알 수 있을까?(Parameter Distribution) - Variational Inference(변분추론) 활용
  
  KL(Kullback-Leibler) Divergence를 통해 두 확률분포(임의로 가정한 분포와 근사치한 분포)의 차이를 계산하고, 이를 최소화시키는 파라미터를 구할 수 있음
  ```
    Kullback-Leibler Divergence(두 확률분포의 차이를 계산)
    
    ![image](https://user-images.githubusercontent.com/108987773/218925492-fbd93bcf-48a0-456a-90cc-fdc7f3ab0497.png)

### Monte Carlo Dropout (MC Dropout)
- 가중치 q(w)를 추정하고자 Monte Carlo Dropout 을 사용함
- 네트워크 가중치(q(w))를 통해 Dropout효과로 동일한 픽셀에서 T개(Dropout Step)의 다른 예측의 분산을 계산하는 예측 레이블의 불확실성을 추정할 수 있음 
- Pixel-wise Uncertainty Maps의 정확성은 T(Dropout Step)과 확률인 p(Dropout probability)에 의해 결정
- p(Dropout probability)가 높음 : 네트워크 가중치의 높은 분산 -> 일관된 결과를 만들기 어려움

### CEAL complementary Sample Selection
- 윤곽에서 멀어질수록, 예측의 전반적인 uncertainty의 기여도가 커져야함
- 예측한 segmentation에 대한 distance map을 계산하여 가중치를 부여
  - Distance Map : 윤곽의 가장 가까운 pixel과의 euclidean거리 계산 -> 윤곽에서 먼 pixel에 대한 가치가 boosted됨
- Distance Map * Uncertainty Map -> 윤과과 멀어질수록 더 높은 점수를 얻음
- 결론적으로, uncertainty map에서 더 두꺼운 윤곽선은 더 두껍게 더 얇은선은 더 얇게 만들어줌

## 2. Complementary Sample Selection
  ![image](https://user-images.githubusercontent.com/108987773/218933005-059635bb-f992-4477-83ab-64a39ccd8552.png)
  - Uncertainty Score가 정의됨 
  - 왼쪽 그래프 : Ground Truth를 가지고 평가한 것이라 실제로는 Uncertainty(세로축)만 추정할 수 있음
    - (1) Undetected Melanomas : High certainty를 가졌으나 prediction이 낮음 -> annotation받아야할 1순위(The most informative candidates to be manualy annotated)
    - (2) Highly uncertain samples : annotation받아야할 2순위
    - (3) Certain Samples : Best candidates to be selected as a pseudo-labels
    - (4) Uncertain and wrong predictions : 가장 좋지 않은 case로 active learning을 반복해서 줄어야함
  - 오른쪽 그래프 : Visualization을 Uncertainty에 사영하여 count함
---
# <4> Results and Future work
1. Dataset : ISIC 2017 challenge dataset :: Skin Lesion Analysis towards melanoma detection
  ![image](https://user-images.githubusercontent.com/108987773/218935512-8ff25a7d-f7a8-4a79-a4ca-3e1af20c8314.png)
  
  - Active Learning Scenario를 가정하기 위해 일부만 Ground Truth로 사용하여 초기 network를 학습하는 용도로 사용
  - 나머지 Ground Truth는 human annotator가 제공하는 것처럼 활용

2. 변형
  - Original : 2,000RGB dermoscopy image + binary mask
  - Modified : Gray Scale Image, CNN(Unet) input 에 맞게 사이즈 조절

3. 학습
  - Training set은 Cost-Effective Active Learning 방법론에 근거하여 초기화함
  - Label data를 랜덤으로 추출한 후 나머지는 Label을 지움
  - Data Augmentation 적용
  
    ![image](https://user-images.githubusercontent.com/108987773/218939702-e1fd939f-e00f-45d4-9705-a387d58a6c52.png)
  - Active Learning Loop의 각 반복에서 sample의 선택은 heuristic parameter로 근거함
  - 초기 600개의 sample 학습을 싲가하여 매 반복마다 1000개의 이미지를 Label 하여 학습에 추가함
    - 매 반복마다 제기된 알고리즘
      - melanoma가 없는 이미지 10개
      - uncertainty가 높은 이미지 10개
      - Random 15개
  - 특정 임계값 이상의 confidence score를 가지면 pseudo label 하여 training set에 추가

4. 결과
  - Segmentation 의 정량적 평가는 Dice Coefficient로 계산함
  - 9번의 반복 active learning 후(CNN 2epoch) 74%의 성능을 보임 => (4) region에 여전히 sample 이 





[link]: https://arxiv.org/pdf/1711.09168.pdf
