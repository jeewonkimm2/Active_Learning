# Cost-Effective Active Learning for Melanoma Segmentation

- Reference : [Cost-Effective Active Learning for Melanoma Segmentation][link]
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
  
  ```
  Monte Carlo Dropout 이란?
  
  ...
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
1. Image uncertainty estimation
- Complementary Sample Selection을 위한 active learning은 unlabeled data의 고유한 분포에 근거함








[link]: https://arxiv.org/pdf/1711.09168.pdf
