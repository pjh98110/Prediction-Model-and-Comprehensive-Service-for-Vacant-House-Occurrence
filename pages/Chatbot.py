import streamlit as st
# import openai
import google.generativeai as genai
from streamlit_chat import message
import os
import requests
from streamlit_extras.colored_header import colored_header
import pandas as pd

# 페이지 구성 설정
st.set_page_config(layout="wide")

# openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = st.secrets["secrets"]["GEMINI_API_KEY"]

# 빈집 유형별 통계 데이터 (사분위수 기준)
EMPTY_HOUSE_STATISTICS = {
    '빈집비율_다세대주택': {
        'mean': 0.179611,
        'q1': 0.102201,
        'q2': 0.177146,
        'q3': 0.239515
    },
    '빈집비율_단독주택': {
        'mean': 0.076250,
        'q1': 0.022859,
        'q2': 0.072799,
        'q3': 0.121156
    },
    '빈집비율_비주거용 건물 내 주택': {
        'mean': 0.123273,
        'q1': 0.095512,
        'q2': 0.128073,
        'q3': 0.154104
    },
    '빈집비율_아파트': {
        'mean': 0.104944,
        'q1': 0.045856,
        'q2': 0.092416,
        'q3': 0.147084
    },
    '빈집비율_연립주택': {
        'mean': 0.165430,
        'q1': 0.091503,
        'q2': 0.149351,
        'q3': 0.224900
    }
}

def get_risk_level(house_type, ratio):
    """빈집비율을 기준으로 위험도 등급 반환"""
    if house_type not in EMPTY_HOUSE_STATISTICS:
        return "데이터 없음"
    
    stats = EMPTY_HOUSE_STATISTICS[house_type]
    
    if ratio >= stats['q3']:
        return "위험"
    elif ratio >= stats['q2']:
        return "보통"
    else:
        return "안전"

def format_region_data_for_prompt(region_data, target_data, max_empty_house_type, selected_district, selected_district2):
    """지역 데이터를 프롬프트용으로 포맷팅"""
    
    # 위험도 등급 계산
    risk_level = get_risk_level(max_empty_house_type, target_data[max_empty_house_type])
    
    formatted_data = f"""
**분석 대상 지역**: {selected_district} {selected_district2}

**빈집 현황 분석**:
- 주요 관심 빈집 유형: {max_empty_house_type}
- 해당 유형 빈집비율: {target_data[max_empty_house_type]:.4f}
- 위험도 등급: {risk_level}
- 전국 평균 대비: {target_data[max_empty_house_type] / EMPTY_HOUSE_STATISTICS[max_empty_house_type]['mean']:.2f}배

**전체 빈집비율 현황**:
"""
    
    for house_type, ratio in target_data.items():
        risk = get_risk_level(house_type, ratio)
        formatted_data += f"- {house_type}: {ratio:.4f} ({risk})\n"
    
    formatted_data += "\n**지역 특성 데이터**:\n"
    
    category_names = {
        'E': '환경 (Environment)',
        'S': '사회/인구 (Social)',
        'G': '거버넌스 (Governance)', 
        'EC': '경제 (Economic)',
        'INF': '인프라 (Infrastructure)'
    }
    
    for category, features in region_data.items():
        if features:
            formatted_data += f"\n**{category_names[category]}**:\n"
            for feature, value in features.items():
                formatted_data += f"- {feature}: {value}\n"
    
    return formatted_data

# 고도화된 Gemini 프롬프트 엔지니어링 함수
def advanced_gemini_prompt(user_input, region_data, target_data, max_empty_house_type, selected_district, selected_district2):
    """
    Persona, Role-Playing, Few-shot learning, Chain of Thought를 적용한 고도화된 프롬프트
    """
    
    # 지역 데이터 포맷팅
    region_info = format_region_data_for_prompt(region_data, target_data, max_empty_house_type, selected_district, selected_district2)
    
    base_prompt = f"""
# 🏡 빈집 문제 전문 분석가 AI 어시스턴트

## 🎭 당신의 역할 (Persona & Role-Playing)
당신은 **대한민국 빈집 문제 해결 전문가**로서 다음과 같은 전문성을 가지고 있습니다:
- 15년 이상의 도시계획 및 주택정책 전문 경험
- 빈집 문제 해결을 위한 정책 설계 및 실행 전문가
- 지역별 특성을 고려한 맞춤형 솔루션 제공 능력
- ESG(환경·사회·거버넌스) 관점에서의 종합적 분석 역량

## 📊 분석 대상 지역 정보
{region_info}

## 🎯 분석 방법론 (Chain of Thought)
다음 단계별로 체계적으로 분석하세요:

### 1단계: 현황 진단
- 해당 지역의 빈집 위험도 등급 판정 및 근거 제시
- 전국 평균 대비 상대적 위치 분석
- 주요 원인 요인 식별 (E-S-G-EC-INF 관점)

### 2단계: 원인 분석
- 빈집 발생의 직접적/간접적 원인 분석
- 지역 특성과 빈집 문제의 연관성 규명
- 다른 지역 사례와의 비교 분석

### 3단계: 솔루션 제안
- 단기(1년), 중기(3년), 장기(5년) 로드맵 제시
- 실현 가능한 구체적 정책 방안 제안
- 예상 효과 및 성과 지표 제시

## 🎓 Few-Shot Learning 예시

**예시 1: 경상북도 문경시 (단독주택 빈집 고위험 지역)**
```
진단: 빈집비율_단독주택 0.145 (위험 등급)
원인: 고령화 심화(독거노인가구비율 높음), 청년 유출(청년순이동률 음수), 경제활동 위축
솔루션: 
- 단기: 빈집 안전관리 조례 제정, 빈집 정비 지원사업
- 중기: 청년 정착 지원 프로그램, 농촌 체험 관광 활성화
- 장기: 스마트 농업 육성, 생활 SOC 확충
```

**예시 2: 서울특별시 강북구 (다세대주택 빈집 보통 위험)**
```
진단: 빈집비율_다세대주택 0.165 (보통 등급)
원인: 노후 다세대주택 집중, 재개발 지연, 임대수요 불안정
솔루션:
- 단기: 빈집 리모델링 지원, 임대주택 전환 지원
- 중기: 소규모 재생사업 추진, 커뮤니티 활성화
- 장기: 도시재생 뉴딜사업 연계, 젠트리피케이션 방지 정책
```

## 🎤 답변 스타일 가이드라인
- **친근하고 전문적인 어조** 사용
- **구체적이고 실행 가능한 방안** 제시
- **근거 기반 분석**으로 신뢰성 확보
- **지역 맞춤형 솔루션** 중심
- **시각적 구분**(이모지, 구조화)으로 가독성 향상

## 📈 성과 측정 지표 제안
각 솔루션에 대해 다음과 같은 KPI를 제시하세요:
- 빈집 감소율 목표
- 주민 만족도 지표
- 지역 경제 활성화 수치
- 사회적 비용 절감 효과

## 🔄 지속적 개선 방안
- 정기적 모니터링 체계
- 주민 참여형 관리 방안
- 성과 평가 및 피드백 시스템

---

**사용자 질문**: {user_input}

위의 전문가적 관점과 체계적 분석 방법론을 바탕으로, 해당 지역의 빈집 문제에 대한 종합적이고 실용적인 솔루션을 제공해주세요.
"""
    
    return base_prompt

# 스트림 표시 함수
def stream_display(response, placeholder):
    text = ''
    for chunk in response:
        if parts := chunk.parts:
            if parts_text := parts[0].text:
                text += parts_text
                placeholder.write(text + "▌")
    return text

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = {
        "gpt": [
            {"role": "system", "content": "안녕하세요, GPT를 기반으로 사용자에게 맞춤형 답변을 드립니다."}
        ],
        "gemini": [
            {"role": "model", "parts": [{"text": "안녕하세요! 빈집 문제 해결 전문가 AI입니다. 선택하신 지역의 빈집 현황을 분석하여 맞춤형 솔루션을 제공해드리겠습니다. 🏡"}]}
        ]
    }

# 세션 변수 체크 함수
def check_session_vars():
    required_vars = ['selected_district', 'selected_district2', 'region_data', 'target_data', 'max_empty_house_type']
    missing_vars = []
    
    for var in required_vars:
        if var not in st.session_state or not st.session_state[var]:
            missing_vars.append(var)
    
    if missing_vars:
        st.warning("⚠️ 빈집 분석을 위한 데이터가 준비되지 않았습니다.")
        st.info("🔄 Home 페이지로 돌아가서 '분석실행' 버튼을 먼저 클릭해주세요.")
        
        if st.button("🏠 Home 페이지로 이동"):
            st.switch_page("Home.py")
        
        st.stop()

# 메인 챗봇 선택
selected_chatbot = st.selectbox(
    "원하는 챗봇을 선택하세요.",
    options=["Gemini를 활용한 시군구별 빈집 유형별 솔루션 챗봇"],
    placeholder="챗봇을 선택하세요.",
    help="선택한 LLM 모델에 따라 다른 챗봇을 제공합니다."
)

if selected_chatbot == "Gemini를 활용한 시군구별 빈집 유형별 솔루션 챗봇":
    colored_header(
        label='🏡 Gemini를 활용한 시군구별 빈집 유형별 솔루션 챗봇',
        description=None,
        color_name="blue-70",
    )
    
    # 세션 변수 체크
    check_session_vars()
    
    # 현재 분석 대상 지역 정보 표시
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.info(f"📍 **분석 지역**: {st.session_state.selected_district} {st.session_state.selected_district2}")
            
        with col2:
            risk_level = get_risk_level(st.session_state.max_empty_house_type, 
                                     st.session_state.target_data[st.session_state.max_empty_house_type])
            risk_color = {"위험": "🔴", "보통": "🟡", "안전": "🟢"}
            st.info(f"🎯 **주요 빈집 유형**: {st.session_state.max_empty_house_type}")
            
        with col3:
            st.info(f"{risk_color.get(risk_level, '⚪')} **위험도**: {risk_level}")

    # 사이드바에서 모델의 파라미터 설정
    with st.sidebar:
        st.header("🔧 모델 설정")
        
        # 2025년 최신 Gemini 모델들 추가
        model_name = st.selectbox(
            "모델 선택",
            [
                'gemini-2.5-pro',           # 최신 고성능 모델
                'gemini-2.5-flash',         # 최신 빠른 모델  
                'gemini-2.5-flash-lite',    # 최신 경량 모델
                'gemini-2.0-flash',         # 2.0 세대 빠른 모델
                'gemini-1.5-pro',           # 이전 세대 고성능
                'gemini-1.5-flash'          # 이전 세대 빠른 모델
            ],
            help="최신 Gemini 2.5 모델들이 더 나은 성능을 제공합니다."
        )
        
        st.divider()
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, max_value=1.0, value=0.3, 
            help="생성 결과의 다양성을 조절합니다. 빈집 분석에는 낮은 값(0.2-0.4)을 권장합니다."
        )
        
        max_output_tokens = st.number_input(
            "Max Tokens", 
            min_value=1, value=8192, 
            help="생성되는 텍스트의 최대 길이를 제한합니다."
        )
        
        top_k = st.slider(
            "Top K", 
            min_value=1, value=40, 
            help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다."
        )
        
        top_p = st.slider(
            "Top P", 
            min_value=0.0, max_value=1.0, value=0.95, 
            help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다."
        )
        
        st.divider()
        st.markdown("### 📊 현재 지역 데이터 요약")
        
        # 간단한 데이터 요약 표시
        if st.session_state.target_data:
            max_ratio = max(st.session_state.target_data.values())
            min_ratio = min(st.session_state.target_data.values())
            st.metric("최고 빈집비율", f"{max_ratio:.4f}")
            st.metric("최저 빈집비율", f"{min_ratio:.4f}")

    # 대화 초기화 버튼
    if st.button("🔄 대화 초기화"):
        st.session_state.messages["gemini"] = [
            {"role": "model", "parts": [{"text": "안녕하세요! 빈집 문제 해결 전문가 AI입니다. 선택하신 지역의 빈집 현황을 분석하여 맞춤형 솔루션을 제공해드리겠습니다. 🏡"}]}
        ]
        st.rerun()

    # 이전 메시지 표시
    if "gemini" not in st.session_state.messages:
        st.session_state.messages["gemini"] = [
            {"role": "model", "parts": [{"text": "안녕하세요! 빈집 문제 해결 전문가 AI입니다. 선택하신 지역의 빈집 현황을 분석하여 맞춤형 솔루션을 제공해드리겠습니다. 🏡"}]}
        ]
        
    for msg in st.session_state.messages["gemini"]:
        role = 'human' if msg['role'] == 'user' else 'assistant'
        with st.chat_message(role):
            st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

    # 추천 질문 버튼들
    st.markdown("### 💡 추천 질문")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🏠 우리 지역 빈집 문제 종합 분석"):
            prompt = "우리 지역의 빈집 문제에 대해 종합적으로 분석하고, 위험도 등급의 근거를 자세히 설명해주세요."
            st.session_state.temp_prompt = prompt
            
        if st.button("📋 단기 실행 가능한 정책 방안"):
            prompt = "1년 내에 실행 가능한 구체적인 빈집 문제 해결 방안을 제시해주세요."
            st.session_state.temp_prompt = prompt

    with col2:
        if st.button("🎯 장기 발전 전략 로드맵"):
            prompt = "5년 장기 관점에서 우리 지역의 빈집 문제를 해결하고 지역 발전을 위한 전략을 제시해주세요."
            st.session_state.temp_prompt = prompt
            
        if st.button("💰 예산 및 기대효과 분석"):
            prompt = "제안하신 솔루션들의 예상 예산과 기대효과를 구체적으로 분석해주세요."
            st.session_state.temp_prompt = prompt

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gemini"].append({"role": "user", "parts": [{"text": prompt}]})
        with st.chat_message('human'):
            st.write(prompt)

        # 고도화된 프롬프트 엔지니어링 적용
        enhanced_prompt = advanced_gemini_prompt(
            prompt, 
            st.session_state.region_data, 
            st.session_state.target_data, 
            st.session_state.max_empty_house_type,
            st.session_state.selected_district,
            st.session_state.selected_district2
        )

        # 모델 호출 및 응답 처리
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_k": top_k,
                "top_p": top_p
            }
            
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            chat = model.start_chat(history=[])  # 새로운 채팅 시작 (프롬프트가 길어서)
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            if not text:
                if (content := response.parts) is not None:
                    text = "응답을 처리 중입니다..."
                    placeholder.write(text + "▌")
                    response = chat.send_message(content, stream=True)
                    text = stream_display(response, placeholder)
            placeholder.write(text)

            # 응답 메시지 저장
            st.session_state.messages["gemini"].append({"role": "model", "parts": [{"text": text}]})
            
        except Exception as e:
            st.error(f"❌ Gemini API 요청 중 오류가 발생했습니다: {str(e)}")
            st.info("💡 다음 사항을 확인해보세요:\n- API 키가 올바른지 확인\n- 네트워크 연결 상태 확인\n- 잠시 후 다시 시도")

    # 추천 질문 버튼 클릭 처리
    if hasattr(st.session_state, 'temp_prompt'):
        prompt = st.session_state.temp_prompt
        del st.session_state.temp_prompt
        
        # 위의 사용자 입력 처리와 동일한 로직 실행
        st.session_state.messages["gemini"].append({"role": "user", "parts": [{"text": prompt}]})
        
        enhanced_prompt = advanced_gemini_prompt(
            prompt, 
            st.session_state.region_data, 
            st.session_state.target_data, 
            st.session_state.max_empty_house_type,
            st.session_state.selected_district,
            st.session_state.selected_district2
        )

        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_k": top_k,
                "top_p": top_p
            }
            
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            chat = model.start_chat(history=[])
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            placeholder.write(text)
            st.session_state.messages["gemini"].append({"role": "model", "parts": [{"text": text}]})
            
        except Exception as e:
            st.error(f"❌ API 요청 중 오류가 발생했습니다: {str(e)}")
        
        st.rerun()
