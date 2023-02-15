# Unsupervised Image Anomaly Detection and Segmentation Based on Pre-Trained Feature Mapping

- Reference : [Unsupervised Image Anomaly Detection and Segmentation Based on Pre-Trained Feature Mapping][link]

이미지 내의 이상적인 또는 비정상적인 영역을 식별하는 기술로, 라벨이 지정된 데이터가 필요하지 않음.

이를 위해 미리 학습된 모델을 사용하여 이미지에서 특징 매핑을 추출하고, 비정상적인 패턴을 탐지하고 분리하는 알고리즘을 적용함.

---
# <1> Introduction
- Automatic product quality inspection이 노동비와 inspection 효율을 개선함
- Image anomaly dection task + Image anomaly segmentation task 어려움
  - 이유 : Abnormal data 모으기 어렵고 anomaly에 대한 라벨이 얻기가 어려움
- 최근 Unsupervised image anomaly detection and segmentation에 대해 **Reconstruction based method**와 **Embedding based method**가 각광을 받음
  1. Reconstruction based method : DCNN(Deep Convolutional Neural Network)를 사용하여 이미지를 복원하여 오리지널 이미지와 복원된 이미지의 차이로 정상/이상을 구분함
  2. Embedding based method : 이미지를 embedded feature space에 mapping하여 이상을 구별함
    ```
    임베딩(Embedding)이란?

    각 데이터를 벡터 공간으로 매핑하므로써, 각 데이터 간의 유사성(similarity)을 계산하거나, 분류, 회귀, 군집화 등의 작업에 사용할 수 있음
    이미지의 경우, 고차원 이미지 정보를 저차원으로 변환하면서 필요한 정보를 보존하는 것. 이미지의 저차원적 특성 벡터를 추출해 이미지에 포함된 내용이 무엇인지 나타내는 일정한 지표를 얻음
    ```
    - Patch SVDD method
      ```
      한 클래스 분류, 즉 이상 감지(anomaly detection)에 사용되는 기계 학습 알고리즘. 한 클래스 분류에서는 목표 클래스에 속하는 인스턴스를, 다른 클래스의 레이블이 지정되지 않은 예제를 사용하지 않고, 목표 클래스의 양성 예제만 사용하여 식별. Patch-SVDD는 입력 데이터의 패치(patch)에 대해 전체 데이터셋이 아닌 지역적인 특징을 캡처하기 위해 지원 벡터 데이터 설명(Support Vector Data Description, SVDD) 알고리즘을 적용. SVDD는 고차원 공간에서 목표 클래스를 설명하기 위해 초구(hypersphere)를 사용하며, 인스턴스와 초구 중심간의 거리가 이상 점수로 사용. Patch-SVDD는 SVDD를 입력 데이터의 패치에 적용하여 지역적인 특징을 캡쳐함
      ```
    - SPADE(Sub-Image Anomaly Detection with Deep Pyramid Correspondences) method
      ```
      딥 러닝을 기반으로 한 이미지 이상 탐지 기술 중 하나. 이 기술은 큰 이미지에서 작은 부분 이미지를 추출하여 각 부분 이미지가 정상 이미지와 비교해 얼마나 이상한지를 판단함. 부분 이미지 간의 대응 관계를 학습하는 "Deep pyramid correspondences"를 사용하여 이상 점수를 계산함. 이 기술의 핵심은 "Deep pyramid correspondences"를 사용하여 부분 이미지 간의 대응 관계를 학습하는 것. 이를 통해 이상한 부분 이미지의 위치를 정확히 파악할 수 있으며, 실제로 이상한 부분 이미지의 위치를 찾아내는 정확도가 높아짐. 딥 러닝을 사용하여 Sub-image 이상 탐지 기술을 구현하는 경우, 대규모의 이미지 데이터셋과 딥 러닝 모델을 학습하는 데 필요한 컴퓨팅 리소스가 필요함. 이러한 리소스가 충분한 경우, 이 기술은 매우 높은 정확도로 이상 부분 이미지를 탐지할 수 있음
      ```
    - 위 두가지 모델의 단점 : 시간이 오래 걸림

- PFM(Pre-trained Feature Mapping)

  ![image](https://user-images.githubusercontent.com/108987773/218954744-516dc1c4-d933-40f7-957b-697417188a44.png)
  
  (a) 기존 mapping 방법 : 정상 이미지는 feature space to feature space 맵핑이 가능, 비정상 이미지는 맵핑 불가능
  
  (b) PFM mapping 방법 : Bidirectional 한 방법을 사용
  
    ![image](https://user-images.githubusercontent.com/108987773/218955128-c672792a-8ad7-46cf-8726-42b48f3aa1c5.png)

  - 결론적으로 성능 향상, 시간 단축

- Main Contributions
  1. 비지도 학습 anomaly detection and segmentation
  2. Bidirectional pre-trained feature mapping과 multi-hierarchical bidirectional pre-trained feature mapping 프레임워크가 제안됨
  3. 좋은 성능(ROCAUC - 97.5% anomaly detection, 97.3% anomaly segmentation), 단축된 computing time
---
# <2> Related Work
### 1. Reconstruction-based Methods
- Main Idea : 정상 이미지의 분포에 초점을 맞춰 복원 후 입력 이미지와의 차이를 통한 점수를 냄
  - Deep Auto Encoder(AE)
    1. Vanila AE : 문제점 - 일반화를 잘 해서 anomaly 부분을 놓칠 수 있음
    2. MemAE(Memory-Augmented AutoEncoder) : 정상 데이터를 Encoding할 때 정상 데이터에 대한 memory를 얻은 후 이를 기반으로 해서 정상 데이터를 생성하는 방법
    3. CAVGA(Convolutional adversarial variational auto encoder with guided attention)
      ```
      CAVGA
      
      이 모델은 여러 가지 딥러닝 기술을 결합하여 구성됩니다.
      첫째, Convolutional Autoencoder는 입력 이미지를 압축하고 다시 복원하는 데 사용됩니다. 이를 통해 이미지의 특징을 추출할 수 있습니다.
      둘째, Adversarial Network는 생성된 이미지와 실제 이미지를 구분하는 판별자를 이용하여 생성자를 학습시킵니다. 이렇게 함으로써 생성된 이미지가 실제 이미지와 구분하기 어려울 정도로 실제같은 이미지를 생성하게 됩니다.
      셋째, Variational Autoencoder는 이미지를 잠재 공간(latent space)으로 표현하고, 이를 이용하여 이미지를 생성하고 편집하는 데 사용됩니다. 이를 통해 이미지의 다양한 변화를 자연스럽게 만들 수 있습니다.

      또한, 이 모델은 Guided Attention이라는 기술을 사용하여 이미지를 특정 부분에 초점을 두고 생성하거나 편집할 수 있습니다. 이를 통해 원하는 부분의 이미지를 강조하거나, 배경과의 차이를 줄이는 등 다양한 효과를 얻을 수 있습니다.
      ```
    4. P-Net Framework : 이미지의 특징을 추출하기 위해 CNN을 사용. CNN을 통해 이미지의 특징을 추출하고, 픽셀 수준에서 라벨링을 수행함. 각 픽셀을 개별적으로 분석할 수 있음. 또한 AutoEncoder를 사용하여 이미지를 재구성함. 이미지의 각 픽셀을 개별적으로 분석하고, 이상을 감지하므로 이미지의 특정 부분에서 이상을 발견할 수 있음. 이러한 방식은 픽셀 수준에서 라벨링을 수행하여, 이미지 전체를 한 번에 분석하는 것보다 더 정확한 이상 탐지를 가능하게 함


- 복원시 anomaly 부분이 전체 복원을 방해하려는 단점을 극복하려고 함

### 2. Embedding-based Methods
- Main Idea : 이미지를 embedding space로 맵핑함. 비정상 이미지의 embedding은 sparse하고 normal cluster으로부터 떨어져 있음
  - pre-trained DCNN
    - CNN_Dict method : pretrained ResNet18을 사용하여 이미지 특성을 patch-by-patch별로 추출. 정상 이미지들은 PCA에 의해 분해되며 k-means clustering방법에 의해 클러스터링됨. 테스팅 샘플들의 특성들도 같은 방식으로 분해되며, 중심점과의 평균거리가 anomaly detection과 segmentation의 점수가 됨.
    -  Student-Teacher(ST) framework : Student network와 Teacher network결과 차이를 비교하여 anomalies를 탐색함. Patch-by-Patch anomaly detection과 segmentation 가능.
    -  SPADE method
    -  Patch SVDD method
- embedding을 기반으로한 PFM framework가 pre-trained 특징을 다른 공간으로 bidirectional하게 맵핑하는 과정을 통하여 anomaly detection과 anomaly segmentation에서 좋은 성능, computing time 을 보임
---
# <3> The Proposed PFM Framework

Basic PFM -> Bidirectional PFM -> Multi-Hierarchical Bidirectional PFM으로 제시

### 1. Pre-trained Feature Mapping
  ![image](https://user-images.githubusercontent.com/108987773/219001960-8b55947c-dac3-4395-a4e7-fd910e072348.png)
  - Main Idea : From one pre-trained space 에서 다른 pre-trained space로 맵핑하기 - 정상 이미지는 자유롭게 mapping이 가능하지만, 비정상 이미지는 mapping 불가능
  
  #### **"To construct a maping neural network"**
  - Training : Only 정상 데이터 -> 정상 이미지에 대해서만 사전 훈련된 DCNN 모델 사이의 간격을 메울 수 있음
  - Optimizer : Mean Squared Loss에 의해 최적화됨
  - Anomaly 판단 : Anomaly 이미지가 큰 MSE(Mean Squared Error)를 낼 것임

  - 과정
 
    ![image](https://user-images.githubusercontent.com/108987773/219010989-ec7373fe-13fc-4032-898c-da7eff818bed.png)
    - Fs : Source embedding feature map
    - Ft : Target embedding feature map
    - SNN : Pre-trained source neural network
    - TNN : Pre-trained target neural network

    ![image](https://user-images.githubusercontent.com/108987773/219012323-bc78429f-325c-4050-99f1-25f64c1c6d4e.png),![image](https://user-images.githubusercontent.com/108987773/219012355-eaeed8e1-9a64-4496-ae77-4d6f49506eec.png),![image](https://user-images.githubusercontent.com/108987773/219013489-ef7d5564-58ae-4bf6-be9b-d2911a772941.png)

    - MNN : Feature mapping function
    - Fms : Mapping 후 Feature Map
    - θms는 정상 이미지에 대한 gradient descent method에 의해 optimization됨
  
    ![image](https://user-images.githubusercontent.com/108987773/219013681-67b89485-89de-4d6f-bee7-960a0f576a9b.png)=> Loss Function
    
    -For a testing image
      - anomaly Scoring Map
      
      ![image](https://user-images.githubusercontent.com/108987773/219015069-06a5c882-9351-43fd-8d20-464094077743.png)
      
      - Anomaly Detection
      
      ![image](https://user-images.githubusercontent.com/108987773/219015256-a7186a18-c71a-4c31-a417-d083ad1e6996.png)

      - Anomaly Segmentation
      
      Anomaly Regions이 항상 높은 점수를 받음












[link]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9795121
