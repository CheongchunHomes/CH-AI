
## 실행 순서

### 1. 백엔드 실행
```bash
# CH-BE 프로젝트를 IntelliJ에서 먼저 실행 후 파이썬 실행 작업 시작
```

### 2. 환경 설정

```bash
# 의존성 설치 및 가상환경 생성
uv sync
```


### 3. 서버 실행

```bash
uv run uvicorn main:app --reload --port 8000
```


### 4. swagger url
```
http://localhost:8000/docs
```
