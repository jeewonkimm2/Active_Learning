# Review of Seminar

- Reference : [Active Learning LAB Seminar from DSBA][link]
---
# <1> Definition of Active Learning

- **Active learning** is a subfield of machine learning and, more generally, artificial intelligence.
- Machine이 배우고 싶은 data를 선택하게 함
  - 장점
    1. 적은 학습으로도 좋은 성능을 냄
    2. Annotation Effort를 줄일 수 있음

# <2> Main Concept

<img width="700" alt="Screenshot 2023-02-08 at 10 07 05 AM" src="https://user-images.githubusercontent.com/108987773/217402519-df79bff9-8fef-4a43-8f26-aeb934594060.png">

  - 라벨이 된 데이터로 학습(supervised learning)을 진행하는 것은 데이터를 구축하는 것 자체가 어렵기 때문에 현실적으로 불가능
  - 그러한 이유로, semi-supervised 방법을 이용하게 됨. 사전에 적은 양의 라벨이 된 데이터로 학습시킨 모델을 활용하여 **추론(Inference)** 과정을 진행하게 되는데, 이 과정에서 **annotation** 이 달림.
  - **문제점** : 사전 학습이 충분하게 이루어지지 않아 성능이 떨어짐 -> annotation 품질 보장 X
  - 위와 같은 문제점으로 인하여, 모델의 Inference 가 **uncertain** 할 경우, 인간 annotator에게 **Query** (라벨링)를 요청함
  - 인간 annotator를 거친 후, 모델 재학습
  - 해당 과정이 반복됨
  
    ```
    The key of active learning is how to measure to uncertainty.
    
    모델이 불확실하다고 판단하는 uncertainty를 어떻게 정의할 것인가?
    ```

# <3> Query Strategies for Active Learning
- **Uncertainty**를 바탕으로 Query 를 줌
- **Uncertainty**를 정하는 방법
  - Uncertainty Sampling
  - Query-By-Committee
  - Expected Model Change
  - Core-Set
  - Learning Loss

### 방법① : Uncertainty sampling
- 오래전부터 사용하였지만 간단하고 직관적이라는 특징이 있어 현재까지도 좋은 성능을 냄
- 초기 학습된 모델의 확신도가 가장 낮은 데이터에게 물어보는 방법 : Decision Boundary 근처에 있는 애매한 데이터들을 Query를 통해 확인하고자 하는 방법
- 가장 단순하고, 보편적으로 많이 사용되는 방법
- Random Sampling에 비해 더 좋은 성능을 낼 수 있음

    ![image](https://user-images.githubusercontent.com/108987773/217423634-2e0e4f28-78f8-4a5c-a240-7ae3200c116e.png)
  - Least Confident
  - Margin Sampling
  - Entropy Sampling

#### 1. Least Confident
  - 모델이 예측한 [각 클래스에 속할 확률] 중 [Top1 확률]이 [가장 낮은 데이터]부터 Query를 줌
    ![image](https://user-images.githubusercontent.com/108987773/217423759-ec053980-d479-406d-b596-25bb91d178d1.png)
    - 모든 클래스(Class 1, Class 2, Class 3)의 확률의 합은 1
    - 위의 예시에선, d2의 데이터 샘플이 어떤 class인지 자신이 제일 없음을 알 수 있음 => Query를 주는 우선순위가 가장 높음

#### 2. Margin Sampling
  - 모델이 예측한 [각 클래스에 속할 확률] 중 [Top1, Top2 확률]의 차이가 [가장 적은 데이터]부터 Query를 줌
    ![image](https://user-images.githubusercontent.com/108987773/217424367-99fac0c2-e5be-42b1-9d89-0b0b3c861dc7.png)
    - 위의 예시에선, d2의 데이터 샘플의 [Top1-Top2]가 가장 적음을 알 수 있음 => Query를 주는 우선순위가 가장 높음

#### 3. Entropy Sampling
  - 모델이 예측한 [각 클래스에 속할 확률]을 활용해, [엔트로피가 가장 큰 데이터]부터 Query를 줌
  - Entropy 구하는 식
  
    ![image](https://user-images.githubusercontent.com/108987773/217425209-a5c53625-adce-4cf4-bed1-f144733b4c68.png)

    ![image](https://user-images.githubusercontent.com/108987773/217425048-2d32cf61-8f21-4f53-b104-20305129bd20.png)
    - 위의 예시에선, d2의 데이터 샘플의 Entropy가 가장 높음을 알 수 있음 => Query를 주는 우선순위가 가장 높음

- 3가지 방법의 비교
  - Sampling 방법에 따라, Query의 우선순위가 바뀔수도 있음

### 방법② : Query-By-Committee(QBC)
- C개의 Committee(=Model)의 Disagreement 정도가 높은 데이터부터 Query를 줌
  ![image](https://user-images.githubusercontent.com/108987773/217426899-05bf66de-c8c4-4855-a963-f0c52a35ed42.png)
  - C=3 의 의미는 Labeled Data 를 가지고 3개의 모델에 각 지도학습을 한다는 것
  - 결과적으로 각 3개의 모델의 Decision Boundary가 생김
  
  ![image](https://user-images.githubusercontent.com/108987773/217427769-8db7fcf6-d57c-4bd1-8642-0c10201994b0.png)
  - 각 Decision Boundary를 통합하였을때, 다르게 판단하는 Data Point들을 우선적으로 Query에 줌
  - 위의 경우에는 'Some Committees disagree' 데이터를 Query에 주는데, 이때 각 모델에 의한 disagreement는 2개, 1개의 종류임을 알 수 있음

### 방법③ : Expected Model Change
- 새로운 Data Point에 대해 Label을 해주었을 때, 현재 모델을 가장 크게 변경시키는 데이터부터 Query를 줌
  ![image](https://user-images.githubusercontent.com/108987773/217428652-f0b8bd9e-65a4-4060-9be8-efc92b88fde8.png)
- Expected Gradient Length(EGL) 이용 : 기대 Gradient가 가장 큰 데이터를 선별하여 Query를 줌
- 이론적으로, Gradient 기반의 학습을 진행하는 모든 모델에 대해 적용 가능
  ![image](https://user-images.githubusercontent.com/108987773/217429001-fa8b10dc-e7fb-4405-925c-7ebfbf10556e.png)
  - 어떤 Data부터 Query를 줄 것인지 결정하기 위해 수식을 적용함
  - x 라는 새로운 데이터가 주어졌을 때, yi class에 속할 확률이 최대화되는 x를 찾고 기존에 있던 Data Point 들의 Loss(L)와의 합집합의 Gradient를 구함
  - 학습을 진행하다보며 이전 데이터에 대한 Gradient의 Loss는 0에 가까워지기 때문에, 새로운 Data Point의 Gradient 값만으로 근사시킬 수 있음

### 방법④ : Core-Set
- Unlabeled 데이터 전체를 Cover할 수 있는 Core-Set을 찾아 Query를 줌
- 비교적 최근 방법

  ![image](https://user-images.githubusercontent.com/108987773/217430413-e132d023-ad12-485d-8f4c-775fb6d8b1fb.png)
  - Unlabeled 데이터 중 랜덤한 Data 하나를 뽑고 δ만큼인 반지름 원을 그렸을때, 전체 데이터를 cover할 수 있는 Data Point(파란색 점)들을 뽑음
  - δ를 최소화 할 수 있게 함(Optimization)
  - Core-Set(파란색 점)
  - Core-Set으로부터 Query를 줌

### 방법⑤ : Learning Loss
- Unlabeled 데이터 중 Loss가 가장 큰 데이터부터 Query를 줌

  ![image](https://user-images.githubusercontent.com/108987773/217432878-5e26bca8-b120-458f-8d21-87be485978e3.png)
  - Loss Prediction Module을 활용하여 Loss 예측
  - Loss가 가장 클 것으로 예측 되는 top-k개의 데이터에 대해 Query 요청

- Loss Function : Margin Ranking Loss
  
  ![image](https://user-images.githubusercontent.com/108987773/217433033-69a09826-e5d9-4d83-b12e-3033af9b84a9.png)
  - 문제점 : MSE Loss를 사용하게 되면 Target Task Loss 값(l)은 학습이 진행됨으로써 감소하기 때문에 scale이 변함

  ![image](https://user-images.githubusercontent.com/108987773/217433139-6c789a59-6ed4-4f13-8b69-5907e9238160.png) ![image](https://user-images.githubusercontent.com/108987773/217433181-d5b894e1-2174-4ff9-8b3e-509efc67bdb0.png)
  - 위와 같은 문제를 해결하기 위해 **Margin Ranking Loss**를 사용하게 됨
  - 최종 Loss Function
  
    ![image](https://user-images.githubusercontent.com/108987773/217433480-8e2e5092-873e-42a1-9403-7d5249e752e2.png)
    - Total Loss를 최소화 시키는 방법으로 학습이 진행됨

- 장점
  1. 기존 Active Learning 방법론들보다 높은 성능
  2. 다양한 Task (Regression, Classification 등) 에서 효과적으로 사용할 수 있음




[link]: https://www.youtube.com/watch?v=Gio7MU5nnc4
