import streamlit as st
import pandas as pd
import numpy as np
import random
import os
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, show_pages
from streamlit_extras.colored_header import colored_header
# from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt

from PIL import Image 
from pathlib import Path


# Streamlit의 경우 로컬 환경에서 실행할 경우 터미널 --> (폴더 경로)Streamlit run Home.py로 실행 / 로컬 환경과 스트리밋 웹앱 환경에서 기능의 차이가 일부 있을 수 있음
# 파일 경로를 잘못 설정할 경우 오류가 발생하고 실행이 불가능하므로 파일 경로 수정 필수
# 데이터 파일의 경우 배포된 웹앱 깃허브에서 다운로드 가능함
# 배포된 서버의 성능 문제로, 코드는 주석처리하고 실제 분석 결과와 시각화 된 이미지로 대체함.

# 페이지 구성 설정
st.set_page_config(layout="wide")

show_pages(
    [
        Page("Home.py", "시군구별 빈집의 유형별 빈집 예측 종합서비스", "🏡"),
        # Page("pages/Chatbot.py", "빈집 위험도 보고서 챗봇", "🤖"),
    ]
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

DATA_PATH = "./"
IMAGE_PATH = "./image/"

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding = "cp949")
 

sample = load_data(f"{DATA_PATH}sample_data.csv")


# 이미지 파일 불러오기
img_path1 = f"{IMAGE_PATH}EDA1.png"
img_path2 = f"{IMAGE_PATH}EDA2.png"
img_path3 = f"{IMAGE_PATH}graph1.png"
img_path4 = f"{IMAGE_PATH}scaling.png"
img_path5 = f"{IMAGE_PATH}최적화된_빈집지수_종합분석.png"
img_path6 = f"{IMAGE_PATH}graph2.png"
img_path7 = f"{IMAGE_PATH}graph3.png"
img_path8 = f"{IMAGE_PATH}graph4.png"
img_path9 = f"{IMAGE_PATH}graph5.png"
img_path10 = f"{IMAGE_PATH}graph6.png"

img_path11 = f"{IMAGE_PATH}graph7.png"
img_path12 = f"{IMAGE_PATH}graph8.png"
img_path13 = f"{IMAGE_PATH}빈집비율.png"
img_path14 = f"{IMAGE_PATH}빈집지수.png"
# img_path11 = f"{IMAGE_PATH}graph6.png"


img_path20 = f"{IMAGE_PATH}Models.png"


def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

reset_seeds(42)

# 한글 폰트 설정 함수
def set_korean_font():
    font_path = f"{DATA_PATH}NanumGothic.ttf"  # 폰트 파일 경로

    from matplotlib import font_manager, rc
    font_manager.fontManager.addfont(font_path)
    rc('font', family='NanumGothic')

# 한글 폰트 설정 적용
set_korean_font()


# 세션 변수에 저장
if 'type_of_case' not in st.session_state:
    st.session_state.type_of_case = None

if 'selected_district' not in st.session_state:
    st.session_state.selected_district = "서울특별시"

# if 'selected_day' not in st.session_state:
#     st.session_state.selected_day = datetime.now()

if 'questions' not in st.session_state:
    st.session_state.questions = None

if 'gpt_input' not in st.session_state:
    st.session_state.gpt_input = None

if 'gemini_input' not in st.session_state:
    st.session_state.gemini_input = None   

if 'selected_survey' not in st.session_state:
    st.session_state.selected_survey = []





# 타이틀
colored_header(
    label= '🏡시군구별 빈집 발생 예측 모델과 종합서비스',
    description=None,
    color_name="blue-70",
)



# # [사이드바]
# st.sidebar.markdown(f"""
#             <span style='font-size: 20px;'>
#             <div style=" color: #000000;">
#                 <strong>사용자 정보 입력</strong>
#             </div>
#             """, unsafe_allow_html=True)


# 사이드바에서 지역 선택
# selected_district = st.sidebar.selectbox(
#     "(1) 당신의 지역을 선택하세요:",
#     ('서울특별시', '경기도', '부산광역시', '인천광역시', '충청북도', '충청남도', 
#      '세종특별자치시', '대전광역시', '전북특별자치도', '전라남도', '광주광역시', 
#      '경상북도', '경상남도', '대구광역시', '울산광역시', '강원특별자치도', '제주특별자치도')
# )
# st.session_state.selected_district = selected_district




selected_survey = st.selectbox(
    "사용할 서비스를 선택하세요.",
    options=["🏡시군구별 빈집의 유형별 빈집비율 예측 종합서비스", ], # "시군구별 빈집", 
    placeholder="하나를 선택하세요.",
    help="선택한 모델에 따라 다른 분석 결과를 제공합니다."
)

st.session_state.selected_survey = selected_survey


if selected_survey == "🏡시군구별 빈집의 유형별 빈집비율 예측 종합서비스":

    st.markdown("### [샘플 데이터 확인]")
    st.markdown("출처: MDIS, 통계청(KOSIS), 부동산 통계정보") 
    st.markdown("데이터: 주택, 인구, 교통, 부동산 등 49개의 공공데이터 사용")
    st.dataframe(sample)

    # st.divider()

    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #A7FFEB;
            width: 100%; /
            display: inline-block;
            margin: 0; /
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    run_button = st.button("분석실행", use_container_width=True)

    if run_button:
        col1, col2 = st.columns(2)
        with col1:

            st.markdown("#### [1. 데이터 스케일링]")
            st.image(Image.open(img_path4), caption="(1) 데이터 스케일링 확인",
            width=600,                      
            clamp=False                     
            )
            st.markdown(
            """
            📈 전체 변수 스케일링 통계 요약:

            스케일링 전 통계:
            평균의 범위: -2.1041 ~ 81456324.5095
            표준편차의 범위: 0.0227 ~ 86271871.8500

            스케일링 후 통계:
            평균의 범위: -17.2644 ~ 1.2532
            표준편차의 범위: 0.0872 ~ 124.2160 \n
            ✅ 스케일링 완료: 277개 변수
            스케일링된 데이터 shape: (15660, 304)
            """)


            st.markdown("#### [3. 요인분석, PCA 등 빈집지수 생성]")
            st.image(Image.open(img_path3), caption="(3) 시도별 평균 빈집지수, 사용된 방법론 분포 등",
            width=600,                      
            clamp=False                     
            )

            st.image(Image.open(img_path6), caption="(3) 방법론별 평균 빈집지수",
            width=600,                      
            clamp=False                     
            )

            st.markdown("#### [4. 변수 중요도(Feature) - 빈집비율_합계]")
            st.image(Image.open(img_path9), caption="(4) 빈집비율_합계 - Shap Summary Plot",
            width=600,                      
            clamp=False                     
            )

            st.image(Image.open(img_path10), caption="(4) 빈집비율_합계 - Shap Waterfall Plot (샘플링)",
            width=600,                      
            clamp=False                     
            )

            st.markdown("#### [5. 빈집 예측 ML 모델 비교]")
            st.image(Image.open(img_path20), caption="(5) 전체 ML 모델 성능 비교",
            width=600,                      
            clamp=False                     
            )

            st.markdown("#### [6. 시군구별 빈집비율 시각화]")
            st.image(Image.open(img_path13), caption="(6) 2023년 시군구별 빈집비율_평균",
            width=600,                      
            clamp=False                     
            )
        

        with col2:

            
            st.markdown("#### [2. EDA (탐색적 데이터 분석)]")

            st.image(Image.open(img_path1), caption="(2) 2023년 시도별 평균 빈집비율",
            width=600,                      
            clamp=False                    
            )
            st.image(Image.open(img_path2), caption="(2) 2023년 시도별 평균 빈집비율_단독주택",
            width=600,                      
            clamp=False                    
            )

            st.markdown("#### [3. 요인분석, PCA 등 빈집지수 생성]")
            st.image(Image.open(img_path7), caption="(3) 시도별 빈집비율 가중치 비교",
            width=600,                      
            clamp=False                     
            )

            st.image(Image.open(img_path8), caption="(3) 2019년 경상북도 요인별 분산 설명력",
            width=600,                      
            clamp=False                     
            )


            st.markdown("#### [4. 변수 중요도(Feature) - 빈집비율_합계]")
            st.image(Image.open(img_path11), caption="(4) 빈집비율_연립주택 - Shap Summary Plot",
            width=600,                      
            clamp=False                     
            )

            st.image(Image.open(img_path12), caption="(4) 빈집비율_연립주택 - Shap Waterfall Plot (샘플링)",
            width=600,                      
            clamp=False                     
            )

            st.markdown("#### [6. 시군구별 빈집지수 시각화]")
            st.image(Image.open(img_path14), caption="(6) 2023년 시군구별 빈집지수_평균",
            width=600,                      
            clamp=False                     
            )
