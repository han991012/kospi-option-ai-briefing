# 코스피 옵션 AI 브리핑 HuggingFace Spaces 배포용 이미지
# 11주차 HF 폴백 자산
#
# 빌드:   docker build -t kospi-option-ai-briefing:latest .
# 실행:   docker run -d -p 7860:7860 --env-file .env kospi-option-ai-briefing:latest
# HF Space에서는 .github/workflows/sync-to-hf.yml이 main push 시 자동 동기화한다.

FROM python:3.12-slim

# 시스템 의존성
# - libjpeg62-turbo, libpng16-16: 이미지 라이브러리 (Pillow 의존성)
# - fonts-nanum: PDF 한글 폰트 (reportlab의 register_korean_font가 자동 인식)
# - ca-certificates: HTTPS 통신용
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libpng16-16 \
        fonts-nanum \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (캐시 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 앱 소스
COPY app.py ./

EXPOSE 7860

# 환경변수
# - SPACE_ID: HF Space 환경 감지용 (app.py가 이걸로 0.0.0.0 바인딩 분기)
# - GRADIO_SERVER_NAME / GRADIO_SERVER_PORT: Gradio fallback
# - PYTHONUNBUFFERED: stdout 즉시 출력 (docker logs 실시간)
ENV SPACE_ID=docker-hf \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONUNBUFFERED=1

# 헬스체크
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/').read()" || exit 1

CMD ["python", "app.py"]
