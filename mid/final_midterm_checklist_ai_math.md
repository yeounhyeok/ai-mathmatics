# !!! 인공지능수학 중간고사 최종점검

시험 전 마지막으로 볼 파일.  
새로 배우는 용도가 아니라, **이미 아는 걸 실전에서 안 틀리기 위한 체크리스트**다.

---

## 0. 시험장 운영 원칙

```text
1. 문제를 보자마자 공식부터 쓰지 말고, shape/부호/변수를 먼저 확인한다.
2. 유도 문제는 핵심 과정 3~5줄이라도 반드시 남긴다.
3. 계산 문제는 마지막에 shape 또는 단위가 맞는지 본다.
4. 검토 시간 5분은 무조건 남긴다.
```

검토 우선순위:

```text
1. @ 결과가 1D인지 2D인지
2. broadcasting은 오른쪽부터 비교했는지
3. residual 부호가 내 공식과 일관되는지
4. chain rule에서 x 또는 -1을 한 번 더/덜 곱하지 않았는지
5. MLE 이계도함수 부호가 음수인지
```

---

## 1. NumPy Shape, @, Broadcasting

### @ 연산 핵심

```text
(n,) @ (n,)       -> scalar, shape ()
(1,n) @ (n,)      -> shape (1,)
(m,n) @ (n,)      -> shape (m,)
(m,n) @ (n,1)     -> shape (m,1)
(n,) @ (n,k)      -> shape (k,)
(m,n) @ (n,k)     -> shape (m,k)
```

주의:

```text
(n,)은 행벡터 (1,n)도 아니고 열벡터 (n,1)도 아니다.
NumPy가 @ 연산 중에 임시로 맞춰주고, 결과에서 차원을 제거할 수 있다.
```

### Broadcasting 핵심

```text
오른쪽 차원부터 비교한다.
둘 중 하나가 1이면 가능.
둘이 같으면 가능.
둘 다 1이 아니고 서로 다르면 불가능.
부족한 차원은 왼쪽에 1을 채워 생각한다.
```

자주 나오는 패턴:

```text
(m,) + (m,)       -> (m,)
(m,) + (m,1)      -> (m,m)
(m,1) + (m,)      -> (m,m)
(m,) + (1,m)      -> (1,m)
scalar + anything -> anything
```

최종 실전 루틴:

```text
1. @ 결과 shape를 먼저 적는다.
2. 그 결과와 더해지는 배열 shape를 비교한다.
3. 1D는 필요하면 왼쪽에 1을 붙여 본다.
```

---

## 2. Indexing

### 기본

```text
X[i, j]       -> scalar, shape ()
X[i:j, :]     -> row slice, 2D 유지
X[:, [a,b]]   -> fancy column indexing
X[mask]       -> True인 행 전체 선택
```

### Fancy indexing 함정

```text
X[[0,2], [1,3]]
```

이건 모든 조합이 아니다. 짝지어서 뽑는다.

```text
X[0,1], X[2,3]
결과 shape -> (2,)
```

모든 조합처럼 2D로 뽑고 싶으면:

```text
X[[0,2]][:, [1,3]]
결과 shape -> (2,2)
```

---

## 3. 표준화와 훈련 통계량

표준화:

```text
x' = (x - mu) / std
```

NumPy:

```python
mu = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_norm = (X_train - mu) / std
X_test_norm = (X_test - mu) / std
```

왜 하는가:

```text
특성 스케일이 다르면 손실 등고선이 길쭉한 타원이 된다.
그러면 gradient descent가 지그재그로 움직이거나 수렴이 느려진다.
표준화하면 특성 스케일이 맞춰져 등고선이 원에 가까워지고 학습이 안정적이다.
```

왜 훈련 통계량만 쓰는가:

```text
테스트 데이터의 평균/표준편차를 쓰면 테스트 정보가 학습 과정에 들어간다.
이건 데이터 누수이고, 실제 일반화 성능 평가를 오염시킨다.
```

---

## 4. 선형변환, 편향 트릭

선형변환 조건:

```text
T(x + y) = T(x) + T(y)
T(cx) = cT(x)
```

```text
T(x) = A @ x        -> 선형변환
T(x) = A @ x + b    -> 아핀 변환, 일반적으로 선형변환 아님
```

편향 트릭:

```python
ones = np.ones((m, 1))
X_bias = np.hstack([ones, X])
w_bias = np.concatenate([[b], w])
y_hat = X_bias @ w_bias
```

의미:

```text
편향 b를 별도로 더하지 않고, 입력에 1열을 추가해서 순수한 행렬곱으로 처리한다.
```

---

## 5. 선형회귀, MSE, Gradient Descent

예측:

```text
y_hat = X @ w
```

잔차:

```text
r = y_hat - y
또는
r = y - y_hat
```

둘 다 쓸 수 있지만, gradient 공식과 부호를 일관되게 유지해야 한다.

MSE:

```text
MSE = mean(residual^2)
```

다변수 MSE gradient:

```text
residual = X @ w - y
grad = (2/m) * X.T @ residual
```

반대 부호 잔차를 쓰면:

```text
residual = y - X @ w
grad = -(2/m) * X.T @ residual
```

경사하강법:

```text
w = w - alpha * grad
```

절대 `+` 방향으로 가지 않는다. gradient는 손실이 증가하는 방향이고, 우리는 반대 방향으로 이동한다.

---

## 6. X.T의 의미

shape:

```text
X: (m,n)
w: (n,)
X @ w: (m,)
residual: (m,)
X.T: (n,m)
X.T @ residual: (n,)
```

설명형 답안:

```text
X는 가중치 공간의 벡터 w를 예측 공간의 벡터 Xw로 보낸다.
잔차 r = Xw - y는 예측 공간에 있는 오차 벡터다.
X.T는 이 잔차를 가중치 공간으로 되돌려,
각 가중치가 손실에 얼마나 영향을 주는지를 나타내는 gradient를 만든다.
```

---

## 7. 과적합과 Weight Decay

과적합:

```text
훈련 데이터에는 잘 맞지만 테스트 데이터에는 성능이 나빠지는 현상.
보통 훈련 손실은 낮고 테스트 손실은 높다.
가중치가 지나치게 커지는 경향이 있을 수 있다.
```

Weight Decay:

```text
loss에 lambda * ||w||^2 항을 추가한다.
```

의미:

```text
큰 가중치에 더 큰 페널티를 줘서 모델이 훈련 데이터에 과하게 맞춰지는 것을 완화한다.
```

---

## 8. 미분과 Sigmoid

기본 미분:

```text
d/dx e^x = e^x
d/dx ln x = 1/x
d/dw e^(-w) = -e^(-w)
d/dw ln(w^2) = 2/w
d/dw (wx - y)^2 = 2x(wx - y)
```

Sigmoid:

```text
sigma(z) = 1 / (1 + e^(-z))
sigma'(z) = sigma(z)(1 - sigma(z))
sigma(0) = 1/2
```

킬러 chain rule:

```text
d/dw ln sigma(wx)
= sigma'(wx) * x / sigma(wx)
= sigma(wx)(1 - sigma(wx)) * x / sigma(wx)
= x(1 - sigma(wx))
```

주의:

```text
sigma'(wx)는 sigma(wx)(1 - sigma(wx))이다.
d/dw sigma(wx)는 sigma'(wx) * x이다.
x를 두 번 곱하지 않는다.
```

---

## 9. 확률, 우도, MLE

조건부확률:

```text
P(A|B) = P(A and B) / P(B)
```

독립:

```text
P(A and B) = P(A)P(B)
```

서로소:

```text
P(A and B) = 0
```

독립과 서로소는 다르다. 둘 다 확률이 양수인 사건이면 서로소이면서 독립일 수 없다.

우도 관점:

```text
확률: p를 고정하고 데이터가 나올 확률을 봄
우도: 데이터를 고정하고 p가 얼마나 그럴듯한지 봄
```

Log-likelihood를 쓰는 이유:

```text
1. 곱을 합으로 바꾼다.
2. 미분이 편해진다.
3. 컴퓨터 계산에서 underflow를 줄인다.
```

이항분포 MLE:

```text
L(p) = C(n,k) * p^k * (1-p)^(n-k)
ell(p) = log L(p)
ell'(p) = k/p - (n-k)/(1-p)
ell'(p) = 0 이면 p_hat = k/n
```

이계도함수:

```text
ell''(p) = -k/p^2 - (n-k)/(1-p)^2
```

`p = k/n`, `0 < k < n`이면 음수이므로 최대점이다.

주의:

```text
ln(1-p), (1-p)^(-1)을 미분할 때는 속미분 -1을 잊지 않는다.
```

---

## 10. 코드 빈칸 최종

```python
# matrix multiplication
y_hat = X @ w

# residual and loss
residual = y_hat - y
loss = np.mean(residual ** 2)

# gradient descent
m = len(y)
grad = (2/m) * X.T @ residual
w = w - alpha * grad

# standardization
mu = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_norm = (X_train - mu) / std
X_test_norm = (X_test - mu) / std

# bias trick
ones = np.ones((X_train.shape[0], 1))
X_bias = np.hstack([ones, X_train_norm])

# sigmoid
z = X @ w
sigmoid = 1 / (1 + np.exp(-z))
```

---

## 11. 시험 직전 10분

이것만 보고 들어간다.

```text
1. (m,n) @ (n,) -> (m,)
2. (m,n) @ (n,1) -> (m,1)
3. (m,) + (m,1) -> (m,m)
4. X[[rows], [cols]] -> 짝지은 좌표, 결과 1D
5. grad = (2/m) * X.T @ (Xw - y)
6. w = w - alpha * grad
7. 표준화는 등고선을 원에 가깝게 해서 GD를 안정화
8. 테스트 데이터 통계량 사용 금지: 데이터 누수
9. d/dw ln sigma(wx) = x(1 - sigma(wx))
10. MLE 이항분포 p_hat = k/n
```

마지막 문장:

```text
새 개념을 떠올리려 하지 말고, 문제에서 요구한 shape/부호/변수를 정확히 읽는다.
```
