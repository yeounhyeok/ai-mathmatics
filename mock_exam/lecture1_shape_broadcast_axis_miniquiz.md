# Lecture 1 약점 보완 미니퀴즈

범위: Broadcasting, 1D/2D transpose/reshape, axis=0/1 표준화  
문항 수: 20문항  
권장 시간: 25분  
목표: 즉답 훈련

---

## Part A. Broadcasting shape 판단 10문제

각 연산이 가능한지 O/X로 표시하고, 가능하면 결과 shape를 쓰시오.

### A-1

```text
A: (3, 2)
b: (2,)

A + b = ?
```

### A-2

```text
A: (3, 2)
c: (3,)

A + c = ?
```

### A-3

```text
A: (3, 2)
d: (3, 1)

A + d = ?
```

### A-4

```text
p: (2, 1)
q: (1, 3)

p + q = ?
```

### A-5

```text
X: (4, 3)
mu: (3,)

X - mu = ?
```

### A-6

```text
X: (4, 3)
row_mean: (4,)

X - row_mean = ?
```

### A-7

```text
X: (4, 3)
row_mean_col: (4, 1)

X - row_mean_col = ?
```

### A-8

```text
u: (5,)
v: (1, 5)

u + v = ?
```

### A-9

```text
u: (5,)
v: (5, 1)

u + v = ?
```

### A-10

```text
M: (2, 3, 4)
b: (4,)

M + b = ?
```

---

## Part B. 1D/2D transpose/reshape 5문제

각 코드의 결과 shape를 쓰시오. 가능하면 왜 그런지도 한 문장으로 설명하시오.

### B-1

```python
x = np.array([1, 2, 3])
x.shape
```

### B-2

```python
x = np.array([1, 2, 3])
x.T.shape
```

### B-3

```python
x = np.array([1, 2, 3])
x.reshape(-1, 1).shape
```

### B-4

```python
x = np.array([1, 2, 3])
x.reshape(1, -1).shape
```

### B-5

```python
x = np.array([1, 2, 3])

x.reshape(-1, 1) @ x.reshape(1, -1)
```

결과 shape는?

---

## Part C. axis=0/1 표준화 코드 빈칸 5문제

### C-1

다음 데이터는 4개 샘플, 3개 특성이다.

```python
X = np.array([
    [1, 10, 100],
    [2, 20, 200],
    [3, 30, 300],
    [4, 40, 400]
])
```

특성별 평균을 구하는 코드의 빈칸을 채우시오.

```python
mu = X.mean(axis=___)
```

### C-2

특성별 표준편차를 구하는 코드의 빈칸을 채우시오.

```python
std = X.std(axis=___)
```

### C-3

특성별 Z-score 표준화 코드를 완성하시오.

```python
mu = X.mean(axis=___)
std = X.std(axis=___)
X_norm = (X - ___) / ___
```

### C-4

샘플별 평균을 구하는 코드의 빈칸을 채우시오.

```python
row_mu = X.mean(axis=___)
```

그리고 `row_mu.shape`는?

### C-5

다음 코드는 샘플별 평균을 각 행에서 빼려는 코드이다. Broadcasting이 되도록 빈칸을 채우시오.

```python
row_mu = X.mean(axis=1)
X_centered_by_row = X - row_mu.reshape(___, ___)
```

`row_mu.reshape(___, ___)`의 shape는 무엇이어야 하는가?

