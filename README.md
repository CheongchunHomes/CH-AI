
## 실행 순서

### 1. 환경 설정

```bash
# venv 설정
# 프로젝트 폴더에서 CMD (PowerShell x안됨)
# venv 생성
py -m venv .venv

# (.venv) 환경으로 진입
.\.venv\Scripts\activate

# venv 환경 확인
python -c "import sys; print(sys.executable)"

# 필요한 라이브러리 설치
pip install -r requirements.txt
```


### 2. 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```


### 3. swagger url
```
http://localhost:8000/docs
```
