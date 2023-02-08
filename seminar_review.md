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





[link]: https://www.youtube.com/watch?v=Gio7MU5nnc4
