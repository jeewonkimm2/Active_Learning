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
    - Patch SVDD method :  한 클래스 분류, 즉 이상 감지(anomaly detection)에 사용되는 기계 학습 알고리즘. 한 클래스 분류에서는 목표 클래스에 속하는 인스턴스를, 다른 클래스의 레이블이 지정되지 않은 예제를 사용하지 않고, 목표 클래스의 양성 예제만 사용하여 식별. Patch-SVDD는 입력 데이터의 패치(patch)에 대해 전체 데이터셋이 아닌 지역적인 특징을 캡처하기 위해 지원 벡터 데이터 설명(Support Vector Data Description, SVDD) 알고리즘을 적용. SVDD는 고차원 공간에서 목표 클래스를 설명하기 위해 초구(hypersphere)를 사용하며, 인스턴스와 초구 중심간의 거리가 이상 점수로 사용. Patch-SVDD는 SVDD를 입력 데이터의 패치에 적용하여 지역적인 특징을 캡쳐함.
    - SPADE(Sub-Image Anomaly Detection with Deep Pyramid Correspondences) method : 딥 러닝을 기반으로 한 이미지 이상 탐지 기술 중 하나. 이 기술은 큰 이미지에서 작은 부분 이미지를 추출하여 각 부분 이미지가 정상 이미지와 비교해 얼마나 이상한지를 판단함. 부분 이미지 간의 대응 관계를 학습하는 "Deep pyramid correspondences"를 사용하여 이상 점수를 계산함. 이 기술의 핵심은 "Deep pyramid correspondences"를 사용하여 부분 이미지 간의 대응 관계를 학습하는 것. 이를 통해 이상한 부분 이미지의 위치를 정확히 파악할 수 있으며, 실제로 이상한 부분 이미지의 위치를 찾아내는 정확도가 높아짐. 딥 러닝을 사용하여 Sub-image 이상 탐지 기술을 구현하는 경우, 대규모의 이미지 데이터셋과 딥 러닝 모델을 학습하는 데 필요한 컴퓨팅 리소스가 필요함. 이러한 리소스가 충분한 경우, 이 기술은 매우 높은 정확도로 이상 부분 이미지를 탐지할 수 있음
    - 위 두가지 모델의 단점 : 시간이 오래 걸림





[link]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9795121
