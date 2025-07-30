import streamlit as st
import openai
import google.generativeai as genai
from streamlit_chat import message
import os
import requests
from streamlit_extras.colored_header import colored_header
import pandas as pd

# 페이지 구성 설정
st.set_page_config(layout="wide")

openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = st.secrets["secrets"]["GEMINI_API_KEY"]



# Gemini 프롬프트 엔지니어링 함수
def gemini_prompt(user_input):
    # 프롬프트 엔지니어링 관련 로직
    base_prompt = f"""
    너는 전문적인 '간편식 추천 및 비교'를 제공하는 보고서 챗봇입니다.


    ESG 기반
    
    - 시도: {st.session_state.selected_district}
    **사용자 정보:**

    - 선택된 가공식품 목록: {st.session_state['recommendations']}

    **규칙:**
    1. 위의 사용자 정보를 바탕으로 답변을 작성합니다.
    2. 선택된 가공식품들을 가격, 영양정보(칼로리, 단밸직, 지방, 탄수화물, 당류, 콜레스테롤, 나트륨 등), 알레르기 유발 성분, 건강 측면에서 비교 분석합니다.
    3. 사용자의 선호하는 맛과 알레르기 정보를 고려하여 가장 적합한 제품을 추천합니다.
    4. 추천한 가공식품과 비슷하고 연관된 가공식품도 추가로 2개 추천합니다.
    5. 답변은 친근하고 유쾌한 어조로 작성합니다.


    가공식품 영양성분 예시:
    [비비고] 사골곰탕 500g: 에너지(kcal) 7, 단백질(g) 0.6, 지방(g) 0.48, 탄수화물(g) 0, 당류(g) 0, 나트륨(mg) 264, 콜레스테롤(mg) 0.7,	포화지방산(g)	0.16, 트랜스지방산(g) 0, 1회 섭취참고량 250g, 식품중량 500g, 가격 800원, 고소한 맛 /n
    [비비고] 육개장 500g: 에너지(kcal) 30, 단백질(g) 2.4, 지방(g) 1.2, 탄수화물(g) 2.4, 당류(g) 1, 나트륨(mg) 406, 콜레스테롤(mg) 3,	포화지방산(g)	0.26, 트랜스지방산(g) 0, 1회 섭취참고량 250g, 식품중량 500g, 가격 970원, 매운맛 /n
    [오뚜기] 마포식 차돌된장찌개 500g: 에너지(kcal) 76, 단백질(g) 4.2, 지방(g) 2.4, 탄수화물(g) 9.4, 당류(g) 1.4, 나트륨(mg) 400, 콜레스테롤(mg) 7,	포화지방산(g)	0.7, 트랜스지방산(g) 0.07, 1회 섭취참고량 200g, 식품중량 500g, 가격 2460원, 짠 맛 /n
    
    사용자 입력: {user_input}
    """
    return base_prompt


# Gemini 프롬프트 엔지니어링 함수
def gemini_prompt2(user_input):
    # 프롬프트 엔지니어링 관련 로직
    base_prompt = f"""
    당신은 창의적인 '간편식 레시피 조합'을 추천하는 요리 전문가 챗봇입니다.

    **사용자 정보:**
    - 성별: {st.session_state.selected_gender}
    - 나이: {st.session_state.selected_age}
    - 선호하는 맛: {st.session_state.selected_taste}
    - 알레르기 정보: {st.session_state.selected_allergy}
    - 선택된 가공식품 목록: {st.session_state['recommendations']}

    **규칙:**
    1. 위의 사용자 정보를 고려하여 레시피 조합을 추천합니다.
    2. 선택된 가공식품들을 활용하여 새로운 요리 아이디어를 제공합니다.
    3. 알레르기 정보를 고려하여 안전한 재료만 사용합니다.
    4. 각 레시피에 대한 간단한 조리 방법과 팁을 제공합니다.
    5. 답변은 친근하고 유쾌한 어조로 작성합니다.


    간편식 레시피 예시: 
    마크정식 레시피:
    [**재료 (1인분 기준)**]
    컵라면 스파게티 1개
    컵라면 떡볶이 1개
    소시지 (프랑크 소시지, 비엔나 소시지 등) 2~3개
    치즈 (모짜렐라 치즈, 슬라이스 치즈 등) 적당량

    [**만드는 법**]
    떡볶이 준비: 컵라면 떡볶이를 조리법대로 끓여줍니다.
    스파게티 준비: 컵라면 스파게티를 조리법대로 끓여줍니다.
    소시지 준비: 소시지를 먹기 좋은 크기로 잘라줍니다.
    모두 합치기: 끓여낸 떡볶이와 스파게티, 소시지를 한 그릇에 담고 치즈를 듬뿍 올려줍니다.
    마무리: 전자레인지에 1-2분 돌려 치즈를 녹여주면 완성!

    [**꿀팁**]
    떡볶이 선택: 매운맛을 좋아한다면 불닭볶음면 떡볶이를 추천합니다.
    스파게티 선택: 크림 스파게티를 사용하면 더욱 부드러운 맛을 즐길 수 있습니다.
    치즈 선택: 모짜렐라 치즈 외에도 체다 치즈, 고다 치즈 등 다양한 치즈를 활용해보세요.
    토핑 추가: 김치, 계란, 참치 등을 추가하여 더욱 풍성하게 즐길 수 있습니다.
    
    사용자 입력: {user_input}
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
            {"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}
        ]
    }

# 세션 변수 체크
def check_session_vars():
    required_vars = ['selected_gender', 'selected_age']
    for var in required_vars:
        if var not in st.session_state:
            st.warning("필요한 정보가 없습니다. 처음으로 돌아가서 정보를 입력해 주세요.")
            st.stop()

selected_chatbot = st.selectbox(
    "원하는 챗봇을 선택하세요.",
    options=["GPT를 활용한 간편식 추천 및 비교", "Gemini를 활용한 시군구별, 빈집의 유형별 솔루션 챗봇", "GPT를 활용한 간편식 레시피 조합 추천", "Gemini를 활용한 간편식 레시피 조합 추천"],
    placeholder="챗봇을 선택하세요.",
    help="선택한 LLM 모델에 따라 다른 챗봇을 제공합니다."
)


if selected_chatbot == "Gemini를 활용한 시군구별, 빈집의 유형별 솔루션 챗봇":
    colored_header(
        label='Gemini를 활용한 시군구별, 빈집의 유형별 솔루션 챗봇',
        description=None,
        color_name="blue-70",
    )
    # 세션 변수 체크
    check_session_vars()

    # 사이드바에서 모델의 파라미터 설정
    with st.sidebar:
        st.header("모델 설정")
        model_name = st.selectbox(
            "모델 선택",
            ['gemini-1.5-flash', "gemini-1.5-pro"]
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, help="생성 결과의 다양성을 조절합니다.")
        max_output_tokens = st.number_input("Max Tokens", min_value=1, value=4096, help="생성되는 텍스트의 최대 길이를 제한합니다.")
        top_k = st.slider("Top K", min_value=1, value=40, help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다.")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다.")

    st.button("대화 초기화", on_click=lambda: st.session_state.update({
        "messages": {"gemini": [{"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}]}
    }))

    # 이전 메시지 표시
    if "gemini" not in st.session_state.messages:
        st.session_state.messages["gemini"] = [
            {"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}
        ]
        
    for msg in st.session_state.messages["gemini"]:
        role = 'human' if msg['role'] == 'user' else 'ai'
        with st.chat_message(role):
            st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gemini"].append({"role": "user", "parts": [{"text": prompt}]})
        with st.chat_message('human'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gemini_prompt(prompt)

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
            chat = model.start_chat(history=st.session_state.messages["gemini"])
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("ai"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            if not text:
                if (content := response.parts) is not None:
                    text = "Wait for function calling response..."
                    placeholder.write(text + "▌")
                    response = chat.send_message(content, stream=True)
                    text = stream_display(response, placeholder)
            placeholder.write(text)

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gemini"].append({"role": "model", "parts": [{"text": text}]})
        except Exception as e:
            st.error(f"Gemini API 요청 중 오류가 발생했습니다: {str(e)}")


elif selected_chatbot == "Gemini를 활용한 간편식 레시피 조합 추천":
    colored_header(
        label='Gemini를 활용한 간편식 레시피 조합 추천',
        description=None,
        color_name="blue-70",
    )
    # 세션 변수 체크
    check_session_vars()

    # 사이드바에서 모델의 파라미터 설정
    with st.sidebar:
        st.header("모델 설정")
        model_name = st.selectbox(
            "모델 선택",
            ['gemini-1.5-flash', "gemini-1.5-pro"]
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, help="생성 결과의 다양성을 조절합니다.")
        max_output_tokens = st.number_input("Max Tokens", min_value=1, value=4096, help="생성되는 텍스트의 최대 길이를 제한합니다.")
        top_k = st.slider("Top K", min_value=1, value=40, help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다.")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다.")

    st.button("대화 초기화", on_click=lambda: st.session_state.update({
        "messages": {"gemini": [{"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}]}
    }))

    # 이전 메시지 표시
    if "gemini" not in st.session_state.messages:
        st.session_state.messages["gemini"] = [
            {"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}
        ]
        
    for msg in st.session_state.messages["gemini"]:
        role = 'human' if msg['role'] == 'user' else 'ai'
        with st.chat_message(role):
            st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gemini"].append({"role": "user", "parts": [{"text": prompt}]})
        with st.chat_message('human'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gemini_prompt2(prompt)

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
            chat = model.start_chat(history=st.session_state.messages["gemini"])
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("ai"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            if not text:
                if (content := response.parts) is not None:
                    text = "Wait for function calling response..."
                    placeholder.write(text + "▌")
                    response = chat.send_message(content, stream=True)
                    text = stream_display(response, placeholder)
            placeholder.write(text)

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gemini"].append({"role": "model", "parts": [{"text": text}]})
        except Exception as e:
            st.error(f"Gemini API 요청 중 오류가 발생했습니다: {str(e)}")