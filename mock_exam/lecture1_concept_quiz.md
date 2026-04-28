# Lecture 1 이해도 심화 퀴즈

범위: Lecture 1 - 벡터, 행렬, NumPy, shape, indexing, broadcasting, Z-score 정규화  
문항 수: 5문항  
목표: 계산보다 개념 이해 확인  
권장 시간: 25-30분

---

## 1. 1D 배열과 2D 열벡터의 차이

다음 코드가 있다.

```python
v = np.array([1, 2, 3])
v_col = v.reshape(-1, 1)
W = np.ones((4, 3))
```

다음을 답하시오.

```text
(a) v.shape는?
(b) v.T.shape는?
(c) v_col.shape는?
(d) W @ v의 결과 shape는?
(e) W @ v_col의 결과 shape는?
```

마지막으로, `v.T`와 `v_col.T`의 결과가 왜 다른지 설명하시오.

---

## 2. 행렬곱과 Broadcasting의 차이

다음 두 연산을 비교하시오.

```python
A = np.ones((3, 2))
b = np.array([10, 20])
c = np.array([1, 2, 3])
```

```text
(a) A + b는 가능한가? 가능하다면 결과 shape는?
(b) A + c는 가능한가? 가능하다면 결과 shape는?
(c) A @ b는 가능한가? 가능하다면 결과 shape는?
(d) A @ c는 가능한가? 가능하다면 결과 shape는?
```

그리고 다음 문장을 완성하시오.

```text
Broadcasting은 주로 ______ 연산에서 작은 배열을 큰 배열에 맞춰 확장하는 규칙이고,
행렬곱 @는 ______ 차원이 맞아야 가능한 연산이다.
```

---

## 3. 인덱싱 방식 구분

다음 배열이 있다.

```python
X = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
    [11, 22, 33]
])
```

다음 코드가 각각 어떤 종류의 인덱싱인지 쓰고, 결과의 shape를 구하시오.

```python
(a) X[2, 1]
(b) X[1:3, 1:]
(c) X[X > 50]
(d) X[[0, 3]]
(e) X[:, [2, 0]]
```

추가 질문:  
`X[X > 50]`의 결과가 2D 행렬이 아니라 1D 배열로 나오는 이유를 설명하시오.

---

## 4. Z-score 정규화와 axis

다음 데이터는 3개의 샘플과 2개의 특성으로 이루어져 있다.

```python
X = np.array([
    [80, 10],
    [60,  5],
    [100, 15]
])
```

다음을 답하시오.

```text
(a) 특성별 평균을 구하려면 axis=0인가, axis=1인가?
(b) 샘플별 평균을 구하려면 axis=0인가, axis=1인가?
(c) Z-score 정규화에서 보통 특성별 평균/표준편차를 쓰는 이유는?
(d) X.mean(axis=1)을 사용해 정규화하면 무엇이 잘못되는가?
```

마지막으로, Z-score 정규화가 gradient descent에 도움이 되는 이유를 `손실 등고선`, `스케일 차이`, `수렴`이라는 단어를 포함해 설명하시오.

---

## 5. 코드 빈칸과 개념 설명

다음 코드는 데이터를 특성별로 표준화하고, bias column을 붙인 뒤 예측값을 계산하려는 코드이다.

```python
X_raw = np.array([
    [84.0, 10.0],
    [59.0,  5.0],
    [112.0, 20.0]
])

mean = X_raw.mean(axis=___)
std = X_raw.std(axis=___)
X_norm = (X_raw - mean) / std

ones = np.ones((___, ___))
X_aug = np.hstack([X_norm, ones])

w = np.array([[1.0], [-0.5], [0.0]])
y_hat = X_aug ___ w
```

다음을 답하시오.

```text
(a) 빈칸을 채우시오.
(b) X_norm의 shape는?
(c) ones의 shape는?
(d) X_aug의 shape는?
(e) y_hat의 shape는?
```

추가 질문:  
`np.ones(3)`을 사용하면 왜 `np.ones((3, 1))`을 쓸 때와 의미가 달라지는지 설명하시오.

