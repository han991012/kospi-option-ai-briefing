---
title: KOSPI Option AI Briefing
emoji: 📊
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 코스피 옵션 AI 브리핑

영업일 하나 입력만으로 — 그날의 코스피200 옵션 시장이 한눈에.

## 기능

- KRX 한국거래소 OpenAPI에서 옵션 일별매매정보 실시간 호출
- 콜·풋 거래량, 평균 IV, 미결제약정 등 통계 자동 집계
- GPT-4o-mini 기반 한국어 시황 리포트 자동 생성
- 한글 폰트 임베딩 PDF 다운로드
- AI 챗봇: 옵션 용어·종목명·시장 해석 질문 응답 (현재 조회 데이터 컨텍스트 주입)

## 기술 스택

- UI: Gradio 6.18.0
- LLM: OpenAI GPT-4o-mini
- 데이터 API: KRX 한국거래소 OpenAPI
- 데이터 처리: pandas
- PDF 생성: reportlab (한글 폰트 임베딩)
- 배포: HuggingFace Spaces (Docker)

## 환경변수

본 Space는 다음 환경변수가 필요합니다 (Settings → Variables and secrets에서 등록):

- `OPENAI_API_KEY` — OpenAI API 키
- `KRX_AUTH_KEY` — KRX 한국거래소 OpenAPI 인증키
