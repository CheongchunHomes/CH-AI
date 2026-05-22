
## 실행 순서

### 1. 환경 설정

```bash
# 의존성 설치 및 가상환경 생성
uv sync
```


### 2. 서버 실행

```bash
uv run uvicorn main:app --reload --port 8000
```


### 3. swagger url
```
http://localhost:8000/docs
```
