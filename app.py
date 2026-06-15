"""
코스피 옵션 AI 브리핑
KRX 옵션 일별매매정보를 받아 통계, AI 시황 리포트, PDF, 챗봇을 제공한다.
"""

import os
import time
from datetime import datetime, timedelta

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


# ========== 환경 변수 로드 ==========
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
KRX_AUTH_KEY = os.environ.get("KRX_AUTH_KEY")

if not OPENAI_API_KEY or not KRX_AUTH_KEY:
    print("⚠️ 환경 변수가 누락되었습니다. .env 파일을 확인하세요.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ========== KRX API 호출 ==========

def get_last_business_day():
    d = datetime.now() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def fetch_krx_options(date_str, timeout=30):
    url = "http://data-dbg.krx.co.kr/svc/apis/drv/opt_bydd_trd"
    headers = {"AUTH_KEY": KRX_AUTH_KEY}
    params = {"basDd": date_str}
    
    response = requests.get(url, headers=headers, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    
    if "OutBlock_1" not in data:
        raise Exception(f"KRX 응답 형식 오류: {data}")
    
    df = pd.DataFrame(data["OutBlock_1"])
    
    if df.empty:
        raise Exception(f"{date_str} 데이터가 없습니다. 휴장일이거나 미래 날짜입니다.")
    
    return df


# ========== 데이터 전처리 & 통계 ==========

def preprocess_data(df):
    numeric_columns = [
        "TDD_CLSPRC", "CMPPREVDD_PRC", "TDD_OPNPRC", "TDD_HGPRC",
        "TDD_LWPRC", "IMP_VOLT", "NXTDD_BAS_PRC", "ACC_TRDVOL",
        "ACC_TRDVAL", "ACC_OPNINT_QTY"
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calculate_statistics(df):
    calls = df[df["RGHT_TP_NM"] == "CALL"]
    puts = df[df["RGHT_TP_NM"] == "PUT"]
    
    return {
        "total_volume": int(df["ACC_TRDVOL"].sum()),
        "total_value": int(df["ACC_TRDVAL"].sum()),
        "call": {
            "volume": int(calls["ACC_TRDVOL"].sum()),
            "value": int(calls["ACC_TRDVAL"].sum()),
            "avg_iv": float(calls["IMP_VOLT"].mean()) if not calls["IMP_VOLT"].isna().all() else 0,
            "open_interest": int(calls["ACC_OPNINT_QTY"].sum()),
        },
        "put": {
            "volume": int(puts["ACC_TRDVOL"].sum()),
            "value": int(puts["ACC_TRDVAL"].sum()),
            "avg_iv": float(puts["IMP_VOLT"].mean()) if not puts["IMP_VOLT"].isna().all() else 0,
            "open_interest": int(puts["ACC_OPNINT_QTY"].sum()),
        },
        "top5_call": calls.nlargest(5, "ACC_TRDVOL")[["ISU_NM", "ACC_TRDVOL"]].reset_index(drop=True),
        "top5_put": puts.nlargest(5, "ACC_TRDVOL")[["ISU_NM", "ACC_TRDVOL"]].reset_index(drop=True),
    }


# ========== AI 리포트 생성 ==========

def build_prompt(date_str, stats):
    formatted_date = f"{date_str[:4]}년 {date_str[4:6]}월 {date_str[6:8]}일"
    
    top5_call_text = "\n".join([
        f"  {idx+1}. {row['ISU_NM']} (거래량 {int(row['ACC_TRDVOL']):,}계약)"
        for idx, row in stats["top5_call"].iterrows()
    ])
    top5_put_text = "\n".join([
        f"  {idx+1}. {row['ISU_NM']} (거래량 {int(row['ACC_TRDVOL']):,}계약)"
        for idx, row in stats["top5_put"].iterrows()
    ])
    
    return f"""당신은 한국 옵션 시장 전문 애널리스트입니다.
아래 KRX 한국거래소 공식 데이터를 바탕으로, 옵션 시장에 익숙하지 않은 일반 투자자도 이해할 수 있는 일일 시황 리포트를 작성해주세요.

[기준일: {formatted_date}]

[시장 전체]
- 총 거래량: {stats['total_volume']:,} 계약
- 총 거래대금: {int(stats['total_value']/1e8):,} 억 원

[콜 옵션 (CALL)]
- 거래량: {stats['call']['volume']:,} 계약
- 거래대금: {int(stats['call']['value']/1e8):,} 억 원
- 평균 내재변동성: {stats['call']['avg_iv']:.2f}%
- 미결제약정: {stats['call']['open_interest']:,} 계약

[풋 옵션 (PUT)]
- 거래량: {stats['put']['volume']:,} 계약
- 거래대금: {int(stats['put']['value']/1e8):,} 억 원
- 평균 내재변동성: {stats['put']['avg_iv']:.2f}%
- 미결제약정: {stats['put']['open_interest']:,} 계약

[거래량 상위 5종목 (콜)]
{top5_call_text}

[거래량 상위 5종목 (풋)]
{top5_put_text}

위 데이터를 바탕으로 다음 사항을 포함한 3~5문단 분량의 시황 리포트를 작성해주세요:

1. 전반적인 시장 분위기 (콜과 풋 중 어느 쪽이 활발했는지)
2. 변동성 스큐 분석 (풋 IV가 콜 IV보다 높은지/낮은지, 어떤 시장 심리를 시사하는지)
3. 거래가 집중된 종목과 그 의미 (행사가 수준 관점)
4. 미결제약정 관점에서 본 포지션 추세

전문 용어는 괄호 안에 간단한 한국어 설명을 덧붙여주세요.
숫자는 데이터에 있는 것만 사용하고, 임의로 만들지 마세요.
마크다운 헤더(###, ##) 사용 자제하고 일반 문단으로 작성해주세요.
"""


def generate_ai_report(date_str, stats):
    prompt = build_prompt(date_str, stats)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 한국 옵션 시장 전문 애널리스트입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7,
            timeout=30
        )
        return response.choices[0].message.content
    
    except Exception as e:
        raise Exception(f"AI 리포트 생성 실패: {e}")


# ========== PDF 생성 ==========

def register_korean_font():
    font_paths = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/MALGUN.TTF",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", path))
                return "KoreanFont"
            except Exception:
                continue
    return "Helvetica"


def generate_pdf(date_str, stats, report_text):
    korean_font = register_korean_font()
    output_path = f"option_briefing_{date_str}.pdf"
    formatted_date = f"{date_str[:4]}년 {date_str[4:6]}월 {date_str[6:8]}일"
    
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'], fontName=korean_font,
                                  fontSize=20, spaceAfter=10, alignment=TA_CENTER, textColor='#0A1929')
    subtitle_style = ParagraphStyle('ST', parent=styles['Normal'], fontName=korean_font,
                                     fontSize=11, spaceAfter=20, alignment=TA_CENTER, textColor='#6B7280')
    heading_style = ParagraphStyle('H', parent=styles['Heading2'], fontName=korean_font,
                                    fontSize=14, spaceBefore=15, spaceAfter=8, textColor='#0A1929')
    body_style = ParagraphStyle('B', parent=styles['Normal'], fontName=korean_font,
                                 fontSize=11, leading=18, spaceAfter=10, alignment=TA_LEFT, textColor='#1F2937')
    footer_style = ParagraphStyle('F', parent=styles['Normal'], fontName=korean_font,
                                   fontSize=8, textColor='#6B7280', alignment=TA_CENTER)
    
    elements = []
    elements.append(Paragraph("코스피 옵션 AI 브리핑", title_style))
    elements.append(Paragraph(f"{formatted_date} 기준 일일 시황 리포트", subtitle_style))
    
    elements.append(Paragraph("■ 시장 요약", heading_style))
    elements.append(Paragraph(
        f"총 거래량: {stats['total_volume']:,} 계약<br/>"
        f"총 거래대금: {int(stats['total_value']/1e8):,} 억 원",
        body_style
    ))
    
    elements.append(Paragraph("■ 콜 옵션 (CALL)", heading_style))
    elements.append(Paragraph(
        f"거래량: {stats['call']['volume']:,} 계약<br/>"
        f"거래대금: {int(stats['call']['value']/1e8):,} 억 원<br/>"
        f"평균 IV: {stats['call']['avg_iv']:.2f}%<br/>"
        f"미결제약정: {stats['call']['open_interest']:,} 계약",
        body_style
    ))
    
    elements.append(Paragraph("■ 풋 옵션 (PUT)", heading_style))
    elements.append(Paragraph(
        f"거래량: {stats['put']['volume']:,} 계약<br/>"
        f"거래대금: {int(stats['put']['value']/1e8):,} 억 원<br/>"
        f"평균 IV: {stats['put']['avg_iv']:.2f}%<br/>"
        f"미결제약정: {stats['put']['open_interest']:,} 계약",
        body_style
    ))
    
    elements.append(Paragraph("■ AI 시황 리포트", heading_style))
    cleaned_report = report_text.replace("###", "").replace("##", "")
    report_html = cleaned_report.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")
    elements.append(Paragraph(report_html, body_style))
    
    elements.append(Spacer(1, 1*cm))
    elements.append(Paragraph(
        "본 리포트는 KRX 한국거래소 공식 데이터와 GPT-4o-mini로 자동 생성되었습니다. "
        "투자 권유가 아니며 참고용으로만 사용하시기 바랍니다.",
        footer_style
    ))
    
    doc.build(elements)
    return output_path


# ========== 챗봇 — KOSPI 옵션 도우미 ==========

DOMAIN_KNOWLEDGE = """KOSPI 옵션 도메인 핵심 지식:

[옵션 종류]
- KOSPI200 정규 옵션: 매월 둘째주 목요일 만기, 승수 25만 원, 종목명 "코스피200"
- 위클리M (Monday): 매주 월요일 만기, 종목명 "코스피위클리M"
- 위클리W (Thursday): 매주 목요일 만기, 종목명 "코스피위클리W"
- 미니 옵션: 정규의 1/5 크기, 승수 5만 원, 종목명 "미니코스피"
- 권리 유형: 콜(CALL, 행사가에 살 권리) / 풋(PUT, 행사가에 팔 권리)

[핵심 지표]
- 종가: 그날 마지막 체결 가격 (포인트 단위)
- 정산가: 거래소가 발표하는 다음날 기준가
- IV (Implied Volatility, 내재변동성): 옵션 가격에서 역산한 미래 변동성 예측치 (%)
  · 평소 15~25%, 시장 불안 시 30%+, 위기 시 60%+
  · 풋 IV > 콜 IV → 하락 헤지 수요 강함 = 변동성 스큐
- 거래량: 그날 새로 체결된 매매 수 (계약 단위)
- 미결제약정 (Open Interest): 아직 청산 안 된 누적 포지션
  · 증가 → 신규 자금 진입, 감소 → 청산
- 행사가 (Strike Price): 옵션 권리 행사 시 적용되는 기준 가격

[종목명 읽는 법]
예: "코스피위클리M C 2606W3 1,417.5 (정규)"
  - 코스피위클리M: 매주 월요일 만기 위클리 옵션
  - C: 콜 (P이면 풋)
  - 2606W3: 2026년 6월 셋째 주 만기
  - 1,417.5: 행사가
  - (정규): 정규 거래시간 (vs 야간)

[승수]
- 정규/위클리: 1포인트 = 25만 원
- 미니: 1포인트 = 5만 원
"""


def build_chatbot_system_prompt(stats_state):
    """챗봇용 시스템 프롬프트를 만든다. 도메인 지식과 (조회됐다면) 현재 데이터 컨텍스트를 포함한다."""
    base = f"""당신은 한국 KOSPI 옵션 시장 전문 도우미입니다.
사용자가 옵션 도메인 용어, 종목명, 시장 해석 등을 묻습니다.
친절하고 정확하게, 일반 투자자도 이해할 수 있는 한국어로 답하세요.

{DOMAIN_KNOWLEDGE}

[중요 규칙]
1. 옵션·파생상품·KOSPI 시장 도메인 질문에만 답하세요.
2. 도메인 외 질문(요리, 게임, 정치 등)은 "옵션 시장 관련 질문에만 답할 수 있습니다"라고 정중히 거절하세요.
3. 확실하지 않은 정보는 "정확히 알지 못합니다"라고 솔직히 답하세요.
4. 데이터에 없는 숫자는 절대 만들어내지 마세요.
5. 답변은 3~5문장 이내로 간결하게. 전문 용어는 괄호로 부연 설명.
"""
    
    if stats_state and isinstance(stats_state, dict) and stats_state.get("loaded"):
        date_str = stats_state.get("date_str", "")
        stats = stats_state.get("stats", {})
        formatted_date = f"{date_str[:4]}년 {date_str[4:6]}월 {date_str[6:8]}일" if date_str else ""
        
        top5_call_text = "\n".join([
            f"  {idx+1}. {row['ISU_NM']} ({int(row['ACC_TRDVOL']):,}계약)"
            for idx, row in stats["top5_call"].iterrows()
        ]) if "top5_call" in stats else ""
        
        top5_put_text = "\n".join([
            f"  {idx+1}. {row['ISU_NM']} ({int(row['ACC_TRDVOL']):,}계약)"
            for idx, row in stats["top5_put"].iterrows()
        ]) if "top5_put" in stats else ""
        
        data_context = f"""

[현재 화면에 조회된 실제 KRX 데이터 — {formatted_date}]
- 총 거래량: {stats['total_volume']:,} 계약
- 총 거래대금: {int(stats['total_value']/1e8):,} 억 원

콜 옵션 (CALL):
- 거래량: {stats['call']['volume']:,} 계약, 거래대금: {int(stats['call']['value']/1e8):,} 억
- 평균 IV: {stats['call']['avg_iv']:.2f}%, 미결제약정: {stats['call']['open_interest']:,}

풋 옵션 (PUT):
- 거래량: {stats['put']['volume']:,} 계약, 거래대금: {int(stats['put']['value']/1e8):,} 억
- 평균 IV: {stats['put']['avg_iv']:.2f}%, 미결제약정: {stats['put']['open_interest']:,}

콜 거래량 TOP 5:
{top5_call_text}

풋 거래량 TOP 5:
{top5_put_text}

[데이터 활용 지침]
- 사용자가 "오늘", "지금", "현재" 같은 표현을 쓰면 위 데이터를 활용해 답하세요.
- "왜 풋이 더 많지?" 같은 질문은 IV 격차, 미결제 추세 등을 함께 해석해주세요.
- 데이터에 없는 종목/숫자는 만들지 마세요.
"""
        base += data_context
    else:
        base += """

[현재 상태]
사용자는 아직 데이터를 조회하지 않았습니다.
일반 옵션 도메인 질문에만 답하고, "조회를 먼저 하시면 오늘 데이터를 함께 분석해드릴 수 있습니다"라고 안내해도 좋습니다.
"""
    
    return base


def chat_respond(message, history, stats_state):
    """챗봇 응답 함수. dict 형식의 history를 받아 OpenAI API를 호출한다."""
    if not message or not message.strip():
        return history, ""
    
    # 시스템 프롬프트 — 도메인 지식과 현재 조회된 데이터를 함께 주입
    system_prompt = build_chatbot_system_prompt(stats_state)
    
    # 최근 5개 메시지만 컨텍스트로 (토큰 절약)
    recent_history = history[-5:] if len(history) > 5 else history
    
    # history: [{"role": "...", "content": "..."}, ...]
    messages = [{"role": "system", "content": system_prompt}]
    for h in recent_history:
        if isinstance(h, dict):
            role = h.get("role", "user")
            content = h.get("content", "")
            if content:
                messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": message})
    
    # OpenAI 호출 (최대 2회 재시도)
    max_retries = 2
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=600,
                temperature=0.5,
                timeout=20,
            )
            answer = response.choices[0].message.content
            
            # dict 형식으로 history 업데이트
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
            return history, ""
        
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(1.5)
                continue
    
    # 모든 재시도 실패
    error_msg = f"⚠️ 응답 생성 실패: {last_error}\n잠시 후 다시 시도해주세요."
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": error_msg},
    ]
    return history, ""


# ========== HTML 렌더링 도우미 ==========

def render_status_html(message, percent, is_error=False):
    if is_error:
        return f"""
        <div style="background: linear-gradient(135deg, #1A1414 0%, #231818 100%); padding: 18px 24px; border-radius: 8px; border-left: 3px solid #D4AF37; margin: 16px 0; box-shadow: 0 2px 8px rgba(239, 68, 68, 0.08);">
            <p style="margin: 0; color: #FCA5A5; font-size: 13px; font-weight: 500; letter-spacing: -0.005em;">{message}</p>
        </div>
        """
    return f"""
    <div style="background: #131A26; padding: 18px 24px; border-radius: 8px; border: 0.5px solid #1F2937; margin: 16px 0; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 6px; height: 6px; border-radius: 50%; background: #D4AF37; animation: pulse 1.5s ease-in-out infinite;"></div>
                <span style="font-size: 13px; color: #F1F5F9; font-weight: 500; letter-spacing: -0.005em;">{message}</span>
            </div>
            <span style="font-size: 12px; color: #D4AF37; font-weight: 600; font-variant-numeric: tabular-nums; letter-spacing: 0.05em;">{percent}%</span>
        </div>
        <div style="background: #0B0F17; height: 6px; border-radius: 3px; overflow: hidden; border: 0.5px solid #1F2937;">
            <div style="background: linear-gradient(90deg, #7C5A0A 0%, #D4AF37 25%, #FEF3C7 50%, #D4AF37 75%, #7C5A0A 100%); background-size: 200% 100%; height: 100%; width: {percent}%; transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1); border-radius: 3px; box-shadow: 0 0 12px rgba(252, 211, 77, 0.5); animation: shimmer 2s linear infinite;"></div>
        </div>
        <style>
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; transform: scale(1); }}
                50% {{ opacity: 0.6; transform: scale(1.3); }}
            }}
            @keyframes shimmer {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
        </style>
    </div>
    """


def render_summary_html(date_str, stats):
    formatted_date = f"{date_str[:4]}년 {date_str[4:6]}월 {date_str[6:8]}일"
    
    top5_call_rows = "".join([
        f'<tr><td style="padding: 7px 0; font-size: 12px; color: #F1F5F9; border-bottom: 0.5px solid #1A1F2A;">'
        f'<span style="color: #D4AF37; font-size: 10px; margin-right: 6px;">{idx+1:02d}</span>{row["ISU_NM"]}</td>'
        f'<td style="text-align: right; color: #94A3B8; font-size: 12px; font-variant-numeric: tabular-nums; padding: 7px 0; border-bottom: 0.5px solid #1A1F2A;">{int(row["ACC_TRDVOL"]):,}</td></tr>'
        for idx, (_, row) in enumerate(stats["top5_call"].iterrows())
    ])
    top5_put_rows = "".join([
        f'<tr><td style="padding: 7px 0; font-size: 12px; color: #F1F5F9; border-bottom: 0.5px solid #1A1F2A;">'
        f'<span style="color: #60A5FA; font-size: 10px; margin-right: 6px;">{idx+1:02d}</span>{row["ISU_NM"]}</td>'
        f'<td style="text-align: right; color: #94A3B8; font-size: 12px; font-variant-numeric: tabular-nums; padding: 7px 0; border-bottom: 0.5px solid #1A1F2A;">{int(row["ACC_TRDVOL"]):,}</td></tr>'
        for idx, (_, row) in enumerate(stats["top5_put"].iterrows())
    ])
    
    return f"""
    <div>
        <div style="display: flex; align-items: center; gap: 10px; margin: 24px 0 14px;">
            <div style="width: 4px; height: 4px; border-radius: 50%; background: #D4AF37;"></div>
            <p style="font-size: 11px; letter-spacing: 0.25em; color: #94A3B8; text-transform: uppercase; margin: 0; font-weight: 600;">
                Market Summary
            </p>
            <span style="font-size: 11px; color: #64748B; font-weight: 500;">· {formatted_date}</span>
            <div style="flex: 1; height: 0.5px; background: #1F2937;"></div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 24px;">
            <div style="background: #131A26; padding: 22px 26px; border-radius: 8px; border: 0.5px solid #1F2937; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
                <p style="font-size: 11px; color: #94A3B8; margin: 0 0 8px; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 500;">총 거래량</p>
                <p style="font-family: 'Noto Serif KR', serif; font-size: 32px; font-weight: 500; color: #F1F5F9; margin: 0; line-height: 1.05; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;">
                    {stats['total_volume']:,}<span style="font-family: 'Noto Sans KR', sans-serif; font-size: 13px; color: #94A3B8; margin-left: 8px; font-weight: 400; letter-spacing: 0;">계약</span>
                </p>
            </div>
            <div style="background: #131A26; padding: 22px 26px; border-radius: 8px; border: 0.5px solid #1F2937; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
                <p style="font-size: 11px; color: #94A3B8; margin: 0 0 8px; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 500;">총 거래대금</p>
                <p style="font-family: 'Noto Serif KR', serif; font-size: 32px; font-weight: 500; color: #F1F5F9; margin: 0; line-height: 1.05; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;">
                    {int(stats['total_value']/1e8):,}<span style="font-family: 'Noto Sans KR', sans-serif; font-size: 13px; color: #94A3B8; margin-left: 8px; font-weight: 400; letter-spacing: 0;">억 원</span>
                </p>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; gap: 10px; margin: 24px 0 14px;">
            <div style="width: 4px; height: 4px; border-radius: 50%; background: #D4AF37;"></div>
            <p style="font-size: 11px; letter-spacing: 0.25em; color: #94A3B8; text-transform: uppercase; margin: 0; font-weight: 600;">
                Call · Put Breakdown
            </p>
            <div style="flex: 1; height: 0.5px; background: #1F2937;"></div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 24px;">
            <div style="background: linear-gradient(135deg, #1A1414 0%, #231818 100%); padding: 22px 26px; border-radius: 8px; border: 0.5px solid #7F1D1D; box-shadow: 0 1px 3px rgba(239, 68, 68, 0.06);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 0.5px solid #EF4444;">
                    <span style="color: #EF4444; font-size: 14px;">▲</span>
                    <span style="font-size: 13px; font-weight: 600; color: #FCA5A5; letter-spacing: 0.02em;">CALL · 콜 옵션</span>
                </div>
                <table style="width: 100%; font-size: 12.5px; border-collapse: collapse;">
                    <tr><td style="padding: 5px 0; color: #FCA5A5;">거래량</td>
                        <td style="padding: 5px 0; text-align: right; color: #FCA5A5; font-weight: 500; font-variant-numeric: tabular-nums;">{stats['call']['volume']:,}<span style="font-size: 10.5px; color: #FCA5A5; margin-left: 4px; font-weight: 400;">계약</span></td></tr>
                    <tr><td style="padding: 5px 0; color: #FCA5A5;">거래대금</td>
                        <td style="padding: 5px 0; text-align: right; color: #FCA5A5; font-weight: 500; font-variant-numeric: tabular-nums;">{int(stats['call']['value']/1e8):,}<span style="font-size: 10.5px; color: #FCA5A5; margin-left: 4px; font-weight: 400;">억</span></td></tr>
                    <tr><td style="padding: 5px 0; color: #FCA5A5;">평균 IV</td>
                        <td style="padding: 5px 0; text-align: right; color: #FCA5A5; font-weight: 500; font-variant-numeric: tabular-nums;">{stats['call']['avg_iv']:.2f}<span style="font-size: 10.5px; color: #FCA5A5; margin-left: 4px; font-weight: 400;">%</span></td></tr>
                    <tr><td style="padding: 5px 0; color: #FCA5A5;">미결제약정</td>
                        <td style="padding: 5px 0; text-align: right; color: #FCA5A5; font-weight: 500; font-variant-numeric: tabular-nums;">{stats['call']['open_interest']:,}</td></tr>
                </table>
            </div>
            <div style="background: linear-gradient(135deg, #0F1726 0%, #141E2F 100%); padding: 22px 26px; border-radius: 8px; border: 0.5px solid #1E3A8A; box-shadow: 0 1px 3px rgba(96, 165, 250, 0.06);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 0.5px solid #60A5FA;">
                    <span style="color: #60A5FA; font-size: 14px;">▼</span>
                    <span style="font-size: 13px; font-weight: 600; color: #93C5FD; letter-spacing: 0.02em;">PUT · 풋 옵션</span>
                </div>
                <table style="width: 100%; font-size: 12.5px; border-collapse: collapse;">
                    <tr><td style="padding: 5px 0; color: #93C5FD;">거래량</td>
                        <td style="padding: 5px 0; text-align: right; color: #93C5FD; font-weight: 500; font-variant-numeric: tabular-nums;">{stats['put']['volume']:,}<span style="font-size: 10.5px; color: #93C5FD; margin-left: 4px; font-weight: 400;">계약</span></td></tr>
                    <tr><td style="padding: 5px 0; color: #93C5FD;">거래대금</td>
                        <td style="padding: 5px 0; text-align: right; color: #93C5FD; font-weight: 500; font-variant-numeric: tabular-nums;">{int(stats['put']['value']/1e8):,}<span style="font-size: 10.5px; color: #93C5FD; margin-left: 4px; font-weight: 400;">억</span></td></tr>
                    <tr><td style="padding: 5px 0; color: #93C5FD;">평균 IV</td>
                        <td style="padding: 5px 0; text-align: right; color: #93C5FD; font-weight: 500; font-variant-numeric: tabular-nums;">{stats['put']['avg_iv']:.2f}<span style="font-size: 10.5px; color: #93C5FD; margin-left: 4px; font-weight: 400;">%</span></td></tr>
                    <tr><td style="padding: 5px 0; color: #93C5FD;">미결제약정</td>
                        <td style="padding: 5px 0; text-align: right; color: #93C5FD; font-weight: 500; font-variant-numeric: tabular-nums;">{stats['put']['open_interest']:,}</td></tr>
                </table>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; gap: 10px; margin: 24px 0 14px;">
            <div style="width: 4px; height: 4px; border-radius: 50%; background: #D4AF37;"></div>
            <p style="font-size: 11px; letter-spacing: 0.25em; color: #94A3B8; text-transform: uppercase; margin: 0; font-weight: 600;">
                Volume Top 5
            </p>
            <div style="flex: 1; height: 0.5px; background: #1F2937;"></div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 8px;">
            <div style="background: #131A26; padding: 18px 22px; border-radius: 8px; border: 0.5px solid #1F2937; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 0.5px solid #1F2937;">
                    <span style="color: #D4AF37; font-size: 12px;">▲</span>
                    <span style="font-size: 12px; font-weight: 600; color: #FCA5A5; letter-spacing: 0.02em;">CALL TOP 5</span>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    {top5_call_rows}
                </table>
            </div>
            <div style="background: #131A26; padding: 18px 22px; border-radius: 8px; border: 0.5px solid #1F2937; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 0.5px solid #1F2937;">
                    <span style="color: #60A5FA; font-size: 12px;">▼</span>
                    <span style="font-size: 12px; font-weight: 600; color: #93C5FD; letter-spacing: 0.02em;">PUT TOP 5</span>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    {top5_put_rows}
                </table>
            </div>
        </div>
    </div>
    """


def markdown_to_html_paragraphs(text):
    lines = text.split("\n")
    html_parts = []
    current_para = []
    
    def flush_para():
        if current_para:
            content = " ".join(current_para)
            html_parts.append(
                f'<p style="font-family: \'Noto Serif KR\', serif; font-size: 14px; line-height: 1.95; color: #F1F5F9; margin: 0 0 14px; letter-spacing: -0.005em;">{content}</p>'
            )
            current_para.clear()
    
    for line in lines:
        line = line.strip()
        if not line:
            flush_para()
        elif line.startswith("### "):
            flush_para()
            html_parts.append(
                f'<p style="font-family: \'Noto Serif KR\', serif; font-size: 15px; font-weight: 600; color: #F1F5F9; margin: 18px 0 8px; letter-spacing: -0.01em;">{line[4:]}</p>'
            )
        elif line.startswith("## "):
            flush_para()
            html_parts.append(
                f'<p style="font-family: \'Noto Serif KR\', serif; font-size: 16px; font-weight: 600; color: #F1F5F9; margin: 20px 0 8px; letter-spacing: -0.01em;">{line[3:]}</p>'
            )
        elif line.startswith("# "):
            flush_para()
            html_parts.append(
                f'<p style="font-family: \'Noto Serif KR\', serif; font-size: 17px; font-weight: 600; color: #F1F5F9; margin: 22px 0 8px; letter-spacing: -0.01em;">{line[2:]}</p>'
            )
        else:
            current_para.append(line)
    
    flush_para()
    return "".join(html_parts)


def render_report_html(date_str, report_text):
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    report_body = markdown_to_html_paragraphs(report_text)
    
    return f"""
    <div>
        <div style="display: flex; align-items: center; gap: 10px; margin: 28px 0 14px;">
            <div style="width: 4px; height: 4px; border-radius: 50%; background: #D4AF37;"></div>
            <p style="font-size: 11px; letter-spacing: 0.25em; color: #94A3B8; text-transform: uppercase; margin: 0; font-weight: 600;">
                AI Market Commentary
            </p>
            <div style="flex: 1; height: 0.5px; background: #1F2937;"></div>
        </div>
        <div style="background: #131A26; border-left: 3px solid #D4AF37; padding: 26px 30px; border-radius: 8px; border-top: 0.5px solid #1F2937; border-right: 0.5px solid #1F2937; border-bottom: 0.5px solid #1F2937; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 0.5px solid #1F2937;">
                <span style="display: inline-flex; align-items: center; gap: 5px; background: #1A1414; color: #D4AF37; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; letter-spacing: 0.05em;">
                    ✨ GPT-4o-mini
                </span>
                <span style="font-size: 11px; color: #94A3B8; letter-spacing: 0.05em;">{formatted_date} 기준</span>
            </div>
            <div style="font-family: 'Noto Serif KR', serif;">
                {report_body}
            </div>
        </div>
    </div>
    """


# ========== Gradio 메인 함수 ==========

def run_analysis(date_input):
    yield (
        gr.update(value=render_status_html("🔄 조회 준비 중...", 5), visible=True),
        gr.update(visible=False),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
    )
    
    if not date_input or not date_input.strip():
        date_str = get_last_business_day()
    else:
        date_str = date_input.replace("-", "").replace("/", "").strip()
        if len(date_str) != 8 or not date_str.isdigit():
            yield (
                gr.update(
                    value=render_status_html(
                        "❌ 날짜 형식이 잘못되었습니다. YYYY-MM-DD 또는 YYYYMMDD 형태로 입력하세요.",
                        0, is_error=True
                    ),
                    visible=True,
                ),
                gr.update(visible=False),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            )
            return
    
    try:
        # KRX 데이터 호출 — 응답 지연 시 자동 재시도 (최대 3회, timeout 점진 증가)
        max_retries = 3
        df = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    status_msg = "📡 KRX 한국거래소에서 옵션 데이터 받는 중..."
                else:
                    status_msg = f"⏳ KRX 서버 응답 지연 — 재시도 중 ({attempt + 1}/{max_retries})..."
                
                yield (
                    gr.update(value=render_status_html(status_msg, 20), visible=True),
                    gr.update(visible=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                )
                
                timeout_sec = 30 + (attempt * 15)
                df = fetch_krx_options(date_str, timeout=timeout_sec)
                df = preprocess_data(df)
                break
            
            except requests.exceptions.Timeout:
                last_error = "응답 지연"
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise Exception(
                    f"KRX 서버 응답 지연이 계속됩니다 ({max_retries}회 시도). "
                    "잠시 후 다시 시도해주세요."
                )
            
            except requests.exceptions.ConnectionError:
                last_error = "연결 끊김"
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                raise Exception(
                    f"KRX 서버 연결 실패 ({max_retries}회 시도). "
                    "네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요."
                )
            
            except Exception as e:
                error_str = str(e)
                if any(keyword in error_str for keyword in ["데이터가 없습니다", "휴장일", "응답 형식 오류", "미래 날짜"]):
                    raise
                last_error = error_str
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
        
        yield (
            gr.update(value=render_status_html("📊 옵션 통계 계산 중 (콜·풋·TOP 5)...", 45), visible=True),
            gr.update(visible=False),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
        stats = calculate_statistics(df)
        summary_html = render_summary_html(date_str, stats)
        
        full_df = df[[
            "RGHT_TP_NM", "ISU_NM",
            "TDD_CLSPRC", "CMPPREVDD_PRC", "IMP_VOLT",
            "NXTDD_BAS_PRC", "ACC_TRDVOL", "ACC_TRDVAL", "ACC_OPNINT_QTY"
        ]].copy()
        full_df["ACC_TRDVAL"] = (full_df["ACC_TRDVAL"] / 1e8).round(1)
        full_df.columns = [
            "권리", "종목명", "종가 (pt)", "대비 (pt)", "IV (%)",
            "정산가 (pt)", "거래량 (계약)", "대금 (억)", "미결제 (계약)"
        ]
        
        yield (
            gr.update(value=render_status_html("🤖 GPT-4o-mini로 AI 시황 리포트 생성 중...", 65), visible=True),
            gr.update(visible=False),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
        report_text = generate_ai_report(date_str, stats)
        report_html = render_report_html(date_str, report_text)
        
        yield (
            gr.update(value=render_status_html("📄 PDF 리포트 생성 중...", 90), visible=True),
            gr.update(visible=False),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
        pdf_path = generate_pdf(date_str, stats, report_text)
        
        # 완료 — 진행 표시 숨기고 결과 표시 + 챗봇용 stats_state 업데이트
        yield (
            gr.update(value="", visible=False),
            gr.update(visible=True),
            summary_html,
            full_df,
            report_html,
            pdf_path,
            {"loaded": True, "date_str": date_str, "stats": stats},
        )
    
    except Exception as e:
        yield (
            gr.update(
                value=render_status_html(f"❌ 오류 발생: {str(e)}", 0, is_error=True),
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )


# ========== Gradio 화면 구성 ==========

CUSTOM_CSS = """
/* ============ 글로벌 폰트 ============ */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&family=Noto+Serif+KR:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Noto Sans KR', 'Inter', -apple-system, sans-serif; }
.serif { font-family: 'Noto Serif KR', serif; }

/* ============ 전체 배경 ============ */
html, body, gradio-app, .gradio-container, .app, .main {
    background: #0B0F17 !important;
    color: #F1F5F9;
}

/* ============ 컨테이너 폭 ============ */
.gradio-container,
gradio-app > div,
gradio-app .main,
.contain,
[class*="container"] {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding-left: 40px !important;
    padding-right: 40px !important;
}

@media (max-width: 768px) {
    .gradio-container,
    gradio-app > div,
    gradio-app .main {
        padding-left: 16px !important;
        padding-right: 16px !important;
    }
}

/* ============ 숫자 폰트 ============ */
.num, table td:not(:nth-child(-n+2)) {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum";
}

/* ============ Gradio 라벨 박스 제거 ============ */
label > span,
.label-wrap,
[data-testid*="label"] {
    background: transparent !important;
    color: #94A3B8 !important;
    font-size: 11px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    padding: 0 !important;
    border: none !important;
}

/* ============ 입력 박스 ============ */
input[type="text"],
.gr-text-input,
textarea {
    background: #131A26 !important;
    border: 0.5px solid #1F2937 !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    color: #F1F5F9 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

input[type="text"]::placeholder,
.gr-text-input::placeholder,
textarea::placeholder {
    color: #64748B !important;
}

input[type="text"]:focus,
.gr-text-input:focus {
    border-color: #D4AF37 !important;
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.15) !important;
    outline: none !important;
}

/* ============ 조회 버튼 ============ */
.gr-button-primary, 
button.primary,
button[variant="primary"] { 
    background: #D4AF37 !important; 
    color: #0B0F17 !important;
    border: 0.5px solid #D4AF37 !important;
    border-radius: 4px !important;
    padding: 12px 32px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 2px 8px rgba(212, 175, 55, 0.25) !important;
    transition: all 0.2s ease !important;
    max-width: 240px !important;
    margin: 0 auto !important;
    display: block !important;
    min-width: 180px !important;
}

.gr-button-primary:hover, 
button.primary:hover {
    background: #E6C24F !important;
    box-shadow: 0 4px 16px rgba(212, 175, 55, 0.4) !important;
    transform: translateY(-1px);
}

/* ============ HTML 표 선 제거 ============ */
.html-container table,
.html-container td,
.html-container tr,
.html-container th,
.gradio-html table,
.gradio-html td,
.gradio-html tr,
.gradio-html th,
.prose table,
.prose td,
.prose tr,
.prose th {
    border: none !important;
    border-collapse: collapse !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* ============ Gradio Dataframe ============ */
.gradio-dataframe,
.svelte-1u9g6yi {
    border: 0.5px solid #1F2937 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}

.gradio-dataframe table thead tr,
.gradio-dataframe table thead th {
    background: #0B0F17 !important;
    color: #94A3B8 !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    padding: 12px 10px !important;
    border-bottom: 0.5px solid #1F2937 !important;
}

.gradio-dataframe table tbody tr {
    border-top: 0.5px solid #1F2937 !important;
    background: #131A26 !important;
    transition: background 0.15s ease;
}

.gradio-dataframe table tbody tr:hover {
    background: #1A2030 !important;
}

.gradio-dataframe table tbody td {
    padding: 10px !important;
    font-size: 12.5px !important;
    color: #F1F5F9 !important;
    font-variant-numeric: tabular-nums;
}

/* ============ 표 정렬: 1·2 컬럼(권리·종목명) 좌측, 나머지(숫자) 가운데 ============ */
.gradio-dataframe table thead th,
.gradio-dataframe table tbody td,
.gradio-dataframe th,
.gradio-dataframe td,
table[class*="dataframe"] th,
table[class*="dataframe"] td {
    text-align: center !important;
}

.gradio-dataframe table thead th:nth-child(-n+2),
.gradio-dataframe table tbody td:nth-child(-n+2),
.gradio-dataframe th:nth-child(-n+2),
.gradio-dataframe td:nth-child(-n+2),
table[class*="dataframe"] th:nth-child(-n+2),
table[class*="dataframe"] td:nth-child(-n+2) {
    text-align: left !important;
}

/* ============ 필터·정렬 메뉴 (floating div) 다크 배경 강제 ============ */
body > div[class*="tippy"],
body > div[class*="floating"],
body > div[class*="popover"],
body > div[class*="dropdown"],
body > div[class*="menu"],
body > div[role="menu"],
body > div[role="listbox"],
body > div[role="tooltip"],
div[data-floating-ui-portal],
.tippy-box,
.tippy-content,
.tippy-arrow,
[data-popper-arrow],
[data-popper-placement] {
    background: #131A26 !important;
    background-color: #131A26 !important;
    color: #F1F5F9 !important;
    border: 0.5px solid #1F2937 !important;
    border-radius: 6px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.7) !important;
    z-index: 99999 !important;
}

body > div[class*="tippy"] *,
body > div[class*="floating"] *,
body > div[class*="popover"] *,
body > div[class*="menu"] *,
body > div[role="menu"] *,
.tippy-box *,
.tippy-content *,
[data-floating-ui-portal] * {
    background: transparent !important;
    background-color: transparent !important;
    color: #F1F5F9 !important;
}

body > div[class*="tippy"] [role="menuitem"]:hover,
body > div[class*="floating"] [role="menuitem"]:hover,
body > div[role="menu"] [role="menuitem"]:hover,
body > div[role="menu"] button:hover,
body > div[role="menu"] a:hover,
body > div[role="menu"] li:hover,
.tippy-content [role="menuitem"]:hover,
[data-floating-ui-portal] [role="menuitem"]:hover {
    background: #1A2030 !important;
    background-color: #1A2030 !important;
    color: #D4AF37 !important;
}

body > div[class*="tippy"] input,
body > div[role="menu"] input,
.tippy-content input,
[data-floating-ui-portal] input {
    background: #0B0F17 !important;
    border: 0.5px solid #1F2937 !important;
    color: #F1F5F9 !important;
    padding: 6px 10px !important;
    border-radius: 4px !important;
}

/* ============ 푸터·기본 UI 정리 ============ */
footer { display: none !important; }
.show-api { display: none !important; }
.built-with { display: none !important; }

/* ============ Gradio Group·Form·Block 회색 배경 제거 (기본) ============ */
.gradio-container .gr-group,
.gradio-container .gradio-group,
.gradio-container .form,
.gradio-container .gradio-row,
.gradio-container .gradio-column,
.gradio-container .block,
.gradio-container [class*="block"][class*="group"],
.gradio-container [class*="group-wrap"],
.gradio-container .grid-wrap,
.gradio-container fieldset,
.gradio-container [data-testid="group"],
gradio-app .form,
gradio-app .gr-group,
gradio-app .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-container .form,
.gradio-container .block {
    padding: 0 !important;
}

/* ============ 결과 영역 wrapper 배경 투명화 ============ */
.dark-result-group,
.dark-result-group > div,
.dark-result-group > div > div,
.dark-result-group > div > div > div,
.dark-result-group > div > div > div > div,
.dark-result-group .form,
.dark-result-group .block,
.dark-result-group .wrap,
.dark-result-group .container,
.dark-result-group [class*="block"],
.dark-result-group [class*="form"],
.dark-result-group [class*="wrap"],
.dark-result-group [class*="panel"],
.dark-result-group > section,
.dark-result-group fieldset {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-color: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ============ 챗봇 다크 럭셔리 ============ */
/* 챗봇 외곽 카드 + 내부 wrapper·bubble-wrap 모두 동일 배경 (입력란과 같은 톤) */
.dark-chatbot,
.dark-chatbot .wrapper,
.dark-chatbot div.wrapper,
.dark-chatbot [class*="wrapper"],
.dark-chatbot .bubble-wrap,
.dark-chatbot div.bubble-wrap,
.dark-chatbot [class*="bubble-wrap"],
div.block.dark-chatbot,
div.block.dark-chatbot > .wrapper,
div.block.dark-chatbot .bubble-wrap {
    background: #131A26 !important;
    background-color: #131A26 !important;
}

/* 외곽 카드 — 보더 + 그림자 + 골드 좌측선 */
.dark-chatbot {
    border: 1px solid #2A3441 !important;
    border-left: 3px solid #D4AF37 !important;
    border-radius: 4px 10px 10px 4px !important;
    box-shadow:
        0 6px 24px rgba(0, 0, 0, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}

/* 내부 wrapper·bubble-wrap은 보더·그림자 없이 (외곽 카드만 카드처럼 보이게) */
.dark-chatbot .wrapper,
.dark-chatbot .bubble-wrap,
.dark-chatbot [class*="wrapper"],
.dark-chatbot [class*="bubble-wrap"] {
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* ============ Gradio 챗봇 기본 액션 버튼 숨김 ============ */
/* 우리가 별도로 만든 "전송", "지우기" 버튼만 사용하기 위함 */
.dark-chatbot button[aria-label*="hare"],
.dark-chatbot button[aria-label*="공유"],
.dark-chatbot button[aria-label*="Copy"],
.dark-chatbot button[aria-label*="copy"],
.dark-chatbot button[aria-label*="복사"],
.dark-chatbot button[aria-label*="Clear"],
.dark-chatbot button[aria-label*="clear"],
.dark-chatbot button[aria-label*="지우"],
.dark-chatbot button[aria-label*="Delete"],
.dark-chatbot button[title*="Copy"],
.dark-chatbot button[title*="Share"],
.dark-chatbot button[title*="Clear"],
.dark-chatbot button[title*="복사"],
.dark-chatbot button[title*="공유"],
.dark-chatbot button[title*="지우"],
.dark-chatbot .icon-button,
.dark-chatbot .icon-buttons,
.dark-chatbot [class*="copy-button"],
.dark-chatbot [class*="share-button"],
.dark-chatbot [class*="action-button"],
.dark-chatbot [class*="message-button"],
.dark-chatbot .actions,
.dark-chatbot .message-buttons,
.dark-chatbot .chatbot-actions {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

.dark-chatbot .message-wrap,
.dark-chatbot [class*="message"] {
    background: transparent !important;
}

.dark-chatbot [data-testid="user"],
.dark-chatbot .user,
.dark-chatbot [class*="user-message"] {
    background: linear-gradient(135deg, #1A2030 0%, #131A26 100%) !important;
    color: #F1F5F9 !important;
    border: 0.5px solid #1F2937 !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
}

.dark-chatbot [data-testid="bot"],
.dark-chatbot .bot,
.dark-chatbot [class*="bot-message"],
.dark-chatbot [class*="assistant"] {
    background: #0B0F17 !important;
    color: #F1F5F9 !important;
    border-left: 3px solid #D4AF37 !important;
    border-top: 0.5px solid #1F2937 !important;
    border-right: 0.5px solid #1F2937 !important;
    border-bottom: 0.5px solid #1F2937 !important;
    border-radius: 4px 8px 8px 8px !important;
    padding: 12px 16px !important;
}

.dark-chatbot * {
    color: #F1F5F9 !important;
}

.dark-chatbot p,
.dark-chatbot li,
.dark-chatbot span {
    line-height: 1.7 !important;
    font-size: 13.5px !important;
}

.dark-chatbot code {
    background: #1A1414 !important;
    color: #D4AF37 !important;
    padding: 2px 6px !important;
    border-radius: 3px !important;
    font-size: 12px !important;
    border: 0.5px solid #1F2937 !important;
}

.dark-chatbot strong {
    color: #D4AF37 !important;
    font-weight: 600 !important;
}

.dark-chat-input textarea {
    background: #131A26 !important;
    border: 0.5px solid #1F2937 !important;
    color: #F1F5F9 !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    min-height: 44px !important;
}

.dark-chat-input textarea:focus {
    border-color: #D4AF37 !important;
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.15) !important;
    outline: none !important;
}

.dark-chat-input textarea::placeholder {
    color: #64748B !important;
}

.dark-chat-send button,
.dark-chat-clear button {
    background: #131A26 !important;
    color: #D4AF37 !important;
    border: 0.5px solid rgba(212, 175, 55, 0.4) !important;
    border-radius: 6px !important;
    padding: 10px 18px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    max-width: none !important;
    min-width: 0 !important;
    margin: 0 !important;
    display: inline-block !important;
}

.dark-chat-send button:hover {
    background: rgba(212, 175, 55, 0.1) !important;
    border-color: #D4AF37 !important;
}

.dark-chat-clear button {
    color: #94A3B8 !important;
    border-color: #1F2937 !important;
}

.dark-chat-clear button:hover {
    background: #1A2030 !important;
    color: #FCA5A5 !important;
}

/* ============ 스크롤바 ============ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0B0F17;
}
::-webkit-scrollbar-thumb {
    background: #1F2937;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94A3B8;
}
"""

HEADER_HTML = """
<div style="text-align: center; padding: 36px 0 8px;">
    <div style="display: flex; align-items: center; gap: 14px; max-width: 720px; margin: 0 auto 22px;">
        <div style="flex: 1; height: 1px; background: linear-gradient(to right, transparent, #D4AF37);"></div>
        <span style="font-size: 11px; letter-spacing: 0.3em; color: #94A3B8; font-weight: 500;">
            DAILY MARKET BRIEFING · VOL. 2026
        </span>
        <div style="flex: 1; height: 1px; background: linear-gradient(to left, transparent, #D4AF37);"></div>
    </div>
    <h1 class="serif" style="font-family: 'Noto Serif KR', serif; font-size: 46px; font-weight: 500; color: #F1F5F9; margin: 0 0 8px; letter-spacing: -0.015em; line-height: 1.1;">
        코스피 옵션 AI 브리핑
    </h1>
    <p style="font-family: 'Noto Serif KR', serif; font-style: italic; color: #94A3B8; font-size: 15px; margin: 0 0 18px; letter-spacing: 0.01em;">
        KOSPI Option Daily Briefing · powered by GPT-4o-mini
    </p>
    <div style="display: flex; align-items: center; gap: 14px; max-width: 280px; margin: 0 auto 28px;">
        <div style="flex: 1; height: 0.5px; background: #1F2937;"></div>
        <span style="color: #D4AF37; font-size: 16px; line-height: 1;">◆</span>
        <div style="flex: 1; height: 0.5px; background: #1F2937;"></div>
    </div>
    
    <!-- 캔들 차트 모티프 -->
    <div style="max-width: 1100px; margin: 0 auto; padding: 0 20px;">
        <svg viewBox="0 0 1200 60" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet" style="width: 100%; height: 56px; display: block;">
            <line x1="50" y1="55" x2="1150" y2="55" stroke="#1F2937" stroke-width="0.5"/>
            <line x1="130" y1="30" x2="130" y2="52" stroke="#EF4444" stroke-width="1"/>
            <rect x="125" y="35" width="10" height="15" fill="#EF4444"/>
            <line x1="240" y1="14" x2="240" y2="52" stroke="#60A5FA" stroke-width="1"/>
            <rect x="235" y="18" width="10" height="30" fill="#60A5FA"/>
            <line x1="350" y1="20" x2="350" y2="50" stroke="#EF4444" stroke-width="1"/>
            <rect x="345" y="25" width="10" height="23" fill="#EF4444"/>
            <line x1="460" y1="34" x2="460" y2="52" stroke="#60A5FA" stroke-width="1"/>
            <rect x="455" y="38" width="10" height="12" fill="#60A5FA"/>
            <line x1="570" y1="12" x2="570" y2="48" stroke="#EF4444" stroke-width="1"/>
            <rect x="565" y="15" width="10" height="30" fill="#EF4444"/>
            <line x1="680" y1="24" x2="680" y2="50" stroke="#60A5FA" stroke-width="1"/>
            <rect x="675" y="28" width="10" height="20" fill="#60A5FA"/>
            <line x1="790" y1="32" x2="790" y2="52" stroke="#EF4444" stroke-width="1"/>
            <rect x="785" y="35" width="10" height="15" fill="#EF4444"/>
            <line x1="900" y1="14" x2="900" y2="48" stroke="#60A5FA" stroke-width="1"/>
            <rect x="895" y="18" width="10" height="27" fill="#60A5FA"/>
            <line x1="1010" y1="24" x2="1010" y2="50" stroke="#EF4444" stroke-width="1"/>
            <rect x="1005" y="28" width="10" height="20" fill="#EF4444"/>
            <line x1="1120" y1="30" x2="1120" y2="52" stroke="#60A5FA" stroke-width="1"/>
            <rect x="1115" y="34" width="10" height="16" fill="#60A5FA"/>
        </svg>
    </div>
</div>
"""

SECTION_LABEL_FULL = """
<div style="display: flex; align-items: center; gap: 10px; margin: 24px 0 12px;">
    <div style="width: 4px; height: 4px; border-radius: 50%; background: #D4AF37;"></div>
    <p style="font-size: 11px; letter-spacing: 0.25em; color: #94A3B8; text-transform: uppercase; margin: 0; font-weight: 600;">
        전체 옵션 데이터
    </p>
    <div style="flex: 1; height: 0.5px; background: #1F2937;"></div>
</div>
"""

SECTION_LABEL_PDF = """
<div style="display: flex; align-items: center; gap: 10px; margin: 24px 0 12px;">
    <div style="width: 4px; height: 4px; border-radius: 50%; background: #D4AF37;"></div>
    <p style="font-size: 11px; letter-spacing: 0.25em; color: #94A3B8; text-transform: uppercase; margin: 0; font-weight: 600;">
        PDF 다운로드
    </p>
    <div style="flex: 1; height: 0.5px; background: #1F2937;"></div>
</div>
"""

SECTION_LABEL_CHATBOT = """
<div style="display: flex; align-items: center; gap: 10px; margin: 36px 0 14px;">
    <div style="width: 4px; height: 4px; border-radius: 50%; background: #D4AF37;"></div>
    <p style="font-size: 11px; letter-spacing: 0.25em; color: #94A3B8; text-transform: uppercase; margin: 0; font-weight: 600;">
        Ask · 옵션 도우미
    </p>
    <span style="font-size: 11px; color: #64748B; font-weight: 500;">· 데이터 기반 AI 챗봇</span>
    <div style="flex: 1; height: 0.5px; background: #1F2937;"></div>
</div>
<div style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(212, 175, 55, 0.02) 100%); border: 0.5px solid rgba(212, 175, 55, 0.2); border-radius: 8px; padding: 14px 20px; margin-bottom: 14px;">
    <p style="margin: 0; color: #CBD5E1; font-size: 12.5px; line-height: 1.6;">
        💬 <span style="color: #D4AF37; font-weight: 600;">KOSPI 옵션 관련 궁금한 점</span>을 물어보세요. 
        용어, 종목명 차이, 시장 해석, 오늘 데이터 분석까지 도와드립니다.
    </p>
</div>
"""


with gr.Blocks(title="코스피 옵션 AI 브리핑") as demo:
    
    gr.HTML(HEADER_HTML)
    
    # 날짜 입력 → 조회 버튼 → 진행 표시 (세로 배치)
    date_input = gr.Textbox(
        label="조회 날짜",
        placeholder="예: 2026-06-12 또는 20260612  (비워두면 어제 영업일)",
        lines=1,
        max_lines=1,
    )
    run_btn = gr.Button("📊 조회", variant="primary")
    
    # 진행 상황 표시 (초기 숨김, 조회 누르면 표시)
    status_output = gr.HTML(visible=False)
    
    # 결과 영역 (조회 전에는 숨김, 조회 후 표시)
    with gr.Group(visible=False, elem_classes="dark-result-group") as result_group:
        
        # 시장 요약 + 콜/풋 + TOP 5
        summary_output = gr.HTML()
        
        # 전체 옵션 데이터
        gr.HTML(SECTION_LABEL_FULL)
        full_table = gr.Dataframe(
            label=None,
            interactive=False,
            wrap=False,
            column_widths=["6%", "26%", "9%", "9%", "8%", "11%", "11%", "10%", "10%"],
            max_height=480,
            elem_id="dark-options-table",
            elem_classes="dark-options-table",
        )
        
        # AI 리포트
        report_output = gr.HTML()
        
        # PDF 다운로드
        gr.HTML(SECTION_LABEL_PDF)
        pdf_file = gr.File(label=None)
        
        # 챗봇 (옵션 도우미)
        gr.HTML(SECTION_LABEL_CHATBOT)
        
        chatbot = gr.Chatbot(
            label=None,
            height=380,
            elem_classes="dark-chatbot",
            value=[
                {"role": "assistant", "content": "안녕하세요. KOSPI 옵션 전문 도우미입니다 ✨\n\n위클리M·W 차이, IV·미결제약정 같은 용어, 또는 **오늘 조회한 데이터에 대한 질문**까지 도와드립니다. 무엇이든 물어보세요."}
            ],
        )
        
        with gr.Row():
            chat_input = gr.Textbox(
                label=None,
                placeholder="예: 위클리M이 뭐예요? / 오늘 거래량 1위가 뭐였어? / 왜 풋 IV가 더 높지?",
                lines=1,
                max_lines=3,
                scale=8,
                elem_classes="dark-chat-input",
            )
            chat_send_btn = gr.Button("전송 ↗", scale=1, elem_classes="dark-chat-send")
            chat_clear_btn = gr.Button("지우기", scale=1, elem_classes="dark-chat-clear")
    
    # 챗봇이 사용할 stats State (세션별 격리)
    stats_state = gr.State(value={"loaded": False, "date_str": "", "stats": {}})
    
    run_btn.click(
        fn=run_analysis,
        inputs=[date_input],
        outputs=[status_output, result_group, summary_output, full_table, report_output, pdf_file, stats_state],
        show_progress="hidden",
    )
    
    # 챗봇 이벤트
    chat_send_btn.click(
        fn=chat_respond,
        inputs=[chat_input, chatbot, stats_state],
        outputs=[chatbot, chat_input],
    )
    chat_input.submit(
        fn=chat_respond,
        inputs=[chat_input, chatbot, stats_state],
        outputs=[chatbot, chat_input],
    )
    chat_clear_btn.click(
        fn=lambda: [{"role": "assistant", "content": "대화를 새로 시작합니다. 다시 질문해주세요 ✨"}],
        inputs=None,
        outputs=[chatbot],
    )
    
    # 페이지 로드 시 JS 주입 (표 가운데 정렬 + 필터 메뉴 다크 톤)
    demo.load(
        fn=lambda: None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            function applyDarkAndAlign() {
                const tables = document.querySelectorAll(
                    '#dark-options-table table, ' +
                    '.dark-options-table table, ' +
                    '[id*="dark-options-table"] table, ' +
                    '[class*="dark-options-table"] table'
                );
                tables.forEach(table => {
                    table.querySelectorAll('tr').forEach(row => {
                        const cells = row.children;
                        for (let i = 0; i < cells.length; i++) {
                            if (i >= 2) {
                                cells[i].style.setProperty('text-align', 'center', 'important');
                            } else {
                                cells[i].style.setProperty('text-align', 'left', 'important');
                            }
                        }
                    });
                });
                
                document.querySelectorAll('div, ul, ol, section').forEach(el => {
                    if (!el || el.dataset.darkApplied === 'true') return;
                    
                    const text = (el.textContent || '').trim();
                    const hasMenuKeyword = (
                        text.includes('오름차순') || 
                        text.includes('내림차순') || 
                        text.includes('정렬 해제') ||
                        text.includes('Clear filter') ||
                        (text.includes('Filter') && text.includes('정렬'))
                    );
                    
                    if (!hasMenuKeyword) return;
                    
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return;
                    if (rect.width > 400 || rect.width < 60) return;
                    if (rect.height > 500) return;
                    
                    el.style.setProperty('background', '#131A26', 'important');
                    el.style.setProperty('background-color', '#131A26', 'important');
                    el.style.setProperty('border', '1px solid #1F2937', 'important');
                    el.style.setProperty('border-radius', '6px', 'important');
                    el.style.setProperty('color', '#F1F5F9', 'important');
                    el.style.setProperty('box-shadow', '0 12px 32px rgba(0,0,0,0.9)', 'important');
                    el.style.setProperty('padding', '6px', 'important');
                    el.style.setProperty('z-index', '99999', 'important');
                    
                    el.querySelectorAll('*').forEach(child => {
                        child.style.setProperty('color', '#F1F5F9', 'important');
                        if (!child.matches('input, textarea')) {
                            child.style.setProperty('background', 'transparent', 'important');
                            child.style.setProperty('background-color', 'transparent', 'important');
                        }
                    });
                    
                    el.querySelectorAll('button, [role="menuitem"], li, a').forEach(item => {
                        if (item.dataset.hoverApplied) return;
                        item.dataset.hoverApplied = 'true';
                        item.addEventListener('mouseenter', () => {
                            item.style.setProperty('background-color', '#1A2030', 'important');
                            item.style.setProperty('color', '#D4AF37', 'important');
                            item.style.setProperty('border-radius', '4px', 'important');
                        });
                        item.addEventListener('mouseleave', () => {
                            item.style.setProperty('background-color', 'transparent', 'important');
                            item.style.setProperty('color', '#F1F5F9', 'important');
                        });
                    });
                    
                    el.querySelectorAll('input, textarea').forEach(inp => {
                        inp.style.setProperty('background', '#0B0F17', 'important');
                        inp.style.setProperty('background-color', '#0B0F17', 'important');
                        inp.style.setProperty('color', '#F1F5F9', 'important');
                        inp.style.setProperty('border', '1px solid #1F2937', 'important');
                        inp.style.setProperty('padding', '6px 10px', 'important');
                        inp.style.setProperty('border-radius', '4px', 'important');
                    });
                    
                    el.dataset.darkApplied = 'true';
                });
            }
            
            applyDarkAndAlign();
            
            // 챗봇 기본 액션 버튼 숨김 (CSS가 못 잡는 경우 대비)
            function hideChatbotDefaultButtons() {
                document.querySelectorAll('.dark-chatbot button').forEach(btn => {
                    // 우리가 만든 "전송 ↗", "지우기" 버튼은 .dark-chat-send / .dark-chat-clear 안에 있어서 제외됨
                    // .dark-chatbot 안의 모든 버튼은 Gradio 기본 액션 버튼
                    const text = (btn.textContent || '').trim();
                    const ariaLabel = btn.getAttribute('aria-label') || '';
                    const title = btn.getAttribute('title') || '';
                    
                    // 텍스트가 없는 아이콘 버튼이거나, share/copy/clear 키워드 포함된 버튼 숨김
                    if (text === '' || 
                        /share|copy|clear|delete|복사|공유|지우/i.test(ariaLabel) ||
                        /share|copy|clear|delete|복사|공유|지우/i.test(title)) {
                        btn.style.setProperty('display', 'none', 'important');
                    }
                });
            }
            hideChatbotDefaultButtons();
            setInterval(hideChatbotDefaultButtons, 500);
            setInterval(applyDarkAndAlign, 300);
            
            return null;
        }
        """,
    )


if __name__ == "__main__":
    # HF Space 환경에서는 SPACE_ID가 설정됨 → 0.0.0.0 바인딩 필요
    is_space = os.environ.get("SPACE_ID") is not None
    
    demo.launch(
        server_name="0.0.0.0" if is_space else "127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=not is_space,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )
