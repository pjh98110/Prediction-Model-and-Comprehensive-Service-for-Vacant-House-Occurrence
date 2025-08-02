# =============================================================================
# inotify 오류 해결을 위한 환경 설정 (코드 최상단에 위치)
# =============================================================================
import os
import sys

# Streamlit file watcher 비활성화 - inotify 오류 해결
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'

import streamlit as st
import pandas as pd
import numpy as np
import random
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header
import requests
import matplotlib.pyplot as plt

from PIL import Image 
from pathlib import Path

# =============================================================================
# Streamlit 추가 설정 (inotify 오류 방지)
# =============================================================================
# 프로그래밍 방식으로 추가 설정 적용
try:
    st.set_option('server.fileWatcherType', 'none')
    st.set_option('server.headless', True)
    st.set_option('browser.gatherUsageStats', False)
except Exception as e:
    # 설정 적용 실패 시에도 앱이 계속 실행되도록 함
    pass

# =============================================================================
# 리소스 정리 함수 (inotify 인스턴스 관리)
# =============================================================================
def cleanup_resources():
    """시스템 리소스 정리 및 inotify 모니터링"""
    try:
        import subprocess
        import logging
        
        # 현재 inotify 사용량 체크
        result = subprocess.run(['find', '/proc/*/fd', '-lname', 'anon_inode:inotify'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.stdout:
            inotify_count = len(result.stdout.strip().split('\n'))
            if inotify_count > 100:  # 임계값 설정
                st.warning("⚠️ 시스템 리소스 사용량이 높습니다. 페이지 새로고침을 권장합니다.")
        
    except Exception:
        # 리소스 체크 실패 시 무시 (앱 실행에 영향 없음)
        pass

# 앱 시작 시 리소스 정리 실행
cleanup_resources()

# Streamlit의 경우 로컬 환경에서 실행할 경우 터미널 --> (폴더 경로)Streamlit run Home.py로 실행
# 파일 경로를 잘못 설정할 경우 오류가 발생하고 실행이 불가능하므로 파일 경로 수정 필수
# 데이터 파일의 경우 배포된 웹앱 깃허브에서 다운로드 가능함
# 배포된 서버의 성능 문제로, 코드는 주석처리하고 실제 분석 결과와 시각화 된 이미지로 대체함.

# 페이지 구성 설정
st.set_page_config(layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Home"

DATA_PATH = "./"
IMAGE_PATH = "./image/"

@st.cache_data
def load_data(file_path):
    """데이터 로드 함수 - 캐싱 적용"""
    try:
        return pd.read_csv(file_path, encoding="cp949")
    except Exception as e:
        st.error(f"데이터 로딩 오류: {e}")
        return pd.DataFrame()  # 빈 DataFrame 반환

# 데이터 로드
sample = load_data(f"{DATA_PATH}sample_df.csv")

# 이미지 파일 경로들 (기존과 동일)
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
img_path20 = f"{IMAGE_PATH}Models.png"

def reset_seeds(seed):
    """시드 설정 함수"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

reset_seeds(42)

# 한글 폰트 설정 함수
def set_korean_font():
    """한글 폰트 설정 - 오류 방지 로직 추가"""
    try:
        font_path = f"{DATA_PATH}NanumGothic.ttf"
        if os.path.exists(font_path):
            from matplotlib import font_manager, rc
            font_manager.fontManager.addfont(font_path)
            rc('font', family='NanumGothic')
        else:
            # 폰트 파일이 없는 경우 기본 설정 사용
            import matplotlib.pyplot as plt
            plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        # 폰트 설정 실패 시에도 앱이 계속 실행되도록 함
        st.warning(f"폰트 설정 중 오류 발생: {e}")

# 한글 폰트 설정 적용
set_korean_font()

# =============================================================================
# 세션 변수 초기화 (기존과 동일)
# =============================================================================
if 'type_of_case' not in st.session_state:
    st.session_state.type_of_case = None

if 'selected_district' not in st.session_state:
    st.session_state.selected_district = "서울특별시"

if 'selected_district2' not in st.session_state:
    st.session_state.selected_district2 = "종로구"

if 'questions' not in st.session_state:
    st.session_state.questions = None

if 'gpt_input' not in st.session_state:
    st.session_state.gpt_input = None

if 'gemini_input' not in st.session_state:
    st.session_state.gemini_input = None   

if 'selected_survey' not in st.session_state:
    st.session_state.selected_survey = []

# 새로운 세션 변수들 - 빈집 분석용 데이터
if 'region_data' not in st.session_state:
    st.session_state.region_data = {}

if 'target_data' not in st.session_state:
    st.session_state.target_data = {}

if 'max_empty_house_type' not in st.session_state:
    st.session_state.max_empty_house_type = None

# =============================================================================
# 기존 함수들 (오류 처리 강화)
# =============================================================================

def get_region_data(selected_district, sample_df):
    """
    선택된 시도와 일치하는 데이터 중 하나를 랜덤하게 선택하여 반환
    오류 처리 강화
    """
    try:
        # 데이터가 비어있는지 확인
        if sample_df.empty:
            st.warning("데이터가 로드되지 않았습니다.")
            return None, None, None
            
        # 시도가 일치하는 데이터 필터링
        filtered_data = sample_df[sample_df['시도'] == selected_district]
        
        if len(filtered_data) == 0:
            st.warning(f"선택한 시도 '{selected_district}'에 해당하는 데이터가 없습니다.")
            return None, None, None
        
        # 랜덤하게 하나 선택
        selected_row = filtered_data.sample(n=1, random_state=42).iloc[0]
        
        # 필요한 컬럼들 정의
        feature_columns = [
            '인구비율',
            '청년순이동률(19~39세) (%)',
            '노후주택비율(%)',
            '주택거래활성도',
            '미분양위험도',
            '의료접근성',
            '청년연앙인구(19~39세) (명)',
            '주택공급밀도',
            '단독주택비율',
            '1인가구비율(%)',
            '독거노인가구비율(%)',
            '보육접근성',
            '문화접근성',
            '생활안전_안전등급',
            '순이동[명]',
            '인구천명당 종사자수',
            '인구밀도',
            '광역교통시설_대중교통/도보_평균접근시간(분)',
            '판매시설_승용차_평균접근시간(분)',
            '재정건전성',
            '교통사고_안전등급',
            '총가구 수(일반가구)',
            '합계출산율(%)',
            '에너지효율'
        ]
        
        target_columns = [
            '빈집비율_다세대주택', 
            '빈집비율_단독주택', 
            '빈집비율_비주거용 건물 내 주택',        
            '빈집비율_아파트', 
            '빈집비율_연립주택'
        ]
        
        # 데이터 추출
        region_features = {}
        target_values = {}
        
        for col in feature_columns:
            if col in selected_row.index:
                region_features[col] = selected_row[col]
        
        for col in target_columns:
            if col in selected_row.index:
                target_values[col] = selected_row[col]
        
        # 가장 높은 빈집비율을 가진 유형 찾기
        if target_values:
            max_empty_house_type = max(target_values, key=target_values.get)
        else:
            max_empty_house_type = None
        
        return region_features, target_values, max_empty_house_type
        
    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}")
        return None, None, None

def categorize_features(region_features):
    """
    특성들을 E, S, G, EC, INF 카테고리로 분류
    """
    categories = {
        'E': {},  # 환경
        'S': {},  # 사회/인구
        'G': {},  # 거버넌스
        'EC': {},  # 경제
        'INF': {}  # 인프라
    }
    
    # E (환경): 에너지효율, 노후주택비율
    e_features = ['에너지효율', '노후주택비율(%)']
    
    # S (사회/인구): 인구 관련 지표들
    s_features = ['인구비율', '청년순이동률(19~39세) (%)', '청년연앙인구(19~39세) (명)', 
                  '단독주택비율', '1인가구비율(%)', '독거노인가구비율(%)', '순이동[명]',
                  '인구밀도', '총가구 수(일반가구)', '합계출산율(%)']
    
    # G (거버넌스): 지역 관리 및 안전
    g_features = ['재정건전성', '생활안전_안전등급', '교통사고_안전등급']
    
    # EC (경제): 경제 활동 관련
    ec_features = ['주택거래활성도', '미분양위험도', '주택공급밀도', '인구천명당 종사자수']
    
    # INF (인프라): 접근성 관련
    inf_features = ['의료접근성', '보육접근성', '문화접근성', 
                    '광역교통시설_대중교통/도보_평균접근시간(분)', '판매시설_승용차_평균접근시간(분)']
    
    # 카테고리별로 분류
    for feature, value in region_features.items():
        if feature in e_features:
            categories['E'][feature] = value
        elif feature in s_features:
            categories['S'][feature] = value
        elif feature in g_features:
            categories['G'][feature] = value
        elif feature in ec_features:
            categories['EC'][feature] = value
        elif feature in inf_features:
            categories['INF'][feature] = value
    
    return categories

# 이미지 로드 함수 (오류 처리 추가)
def load_image_safely(image_path, caption=""):
    """안전한 이미지 로드 함수"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            st.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return None
    except Exception as e:
        st.error(f"이미지 로드 오류: {e}")
        return None

# =============================================================================
# 메인 UI 코드 (기존과 동일)
# =============================================================================

# 타이틀
colored_header(
    label='🏡시군구별 유형별 빈집 예측 모델과 솔루션 챗봇',
    description=None,
    color_name="blue-70",
)

# [사이드바]
st.sidebar.markdown(f"""
            <span style='font-size: 20px;'>
            <div style=" color: #000000;">
                <strong>사용자 지역 선택</strong>
            </div>
            """, unsafe_allow_html=True)

selected_district = st.sidebar.selectbox(
    "(1) 당신의 시도를 선택하세요:",
    ("서울특별시", "경기도", "부산광역시", "인천광역시", 
    "대구광역시", "대전광역시", "울산광역시", "경상북도",
    "경상남도", "전라북도", "전라남도", "강원특별자치도",
    "충청북도", "충청남도", "세종특별자치시", "광주광역시",
    "제주특별자치도")
)
st.session_state.selected_district = selected_district

selected_district2 = st.sidebar.text_input(
    "(2) 당신의 시군구를 입력하세요:",
)
st.session_state.selected_district2 = selected_district2

selected_survey = st.selectbox(
    "사용할 서비스를 선택하세요.",
    options=["🏡시군구별 빈집 발생 예측 모델과 종합서비스", ], 
    placeholder="하나를 선택하세요.",
    help="선택한 모델에 따라 다른 분석 결과를 제공합니다."
)

st.session_state.selected_survey = selected_survey

if selected_survey == "🏡시군구별 빈집 발생 예측 모델과 종합서비스":

    st.markdown("### [샘플 데이터 확인]")
    st.markdown("출처: MDIS, 통계청(KOSIS), 부동산 통계정보") 
    st.markdown("데이터: 주택, 인구, 교통, 부동산 등 49개의 공공데이터 사용")
    
    # 데이터프레임 표시 (오류 처리 추가)
    if not sample.empty:
        st.dataframe(sample)
    else:
        st.warning("샘플 데이터를 로드할 수 없습니다.")

    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #A7FFEB;
            width: 100%; 
            display: inline-block;
            margin: 0; 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    run_button = st.button("분석실행", use_container_width=True)

    if run_button:
        # 리소스 상태 체크
        cleanup_resources()
        
        # 선택된 지역의 데이터 가져오기
        region_features, target_values, max_empty_house_type = get_region_data(selected_district, sample)
        
        if region_features is not None:
            # 세션 상태에 저장
            st.session_state.region_data = categorize_features(region_features)
            st.session_state.target_data = target_values
            st.session_state.max_empty_house_type = max_empty_house_type
            
            # 선택된 데이터 정보 표시
            st.success(f"✅ {selected_district} 지역의 데이터가 선택되었습니다!")
            
            if max_empty_house_type and target_values:
                st.info(f"🎯 가장 높은 빈집비율을 보이는 유형: **{max_empty_house_type}** ({target_values[max_empty_house_type]:.4f})")
            
            # 카테고리별 데이터 미리보기
            with st.expander("📊 선택된 지역 데이터 미리보기"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 빈집비율 데이터:**")
                    for housing_type, ratio in target_values.items():
                        st.write(f"- {housing_type}: {ratio:.4f}")
                
                with col2:
                    st.markdown("**🏘️ 지역 특성 요약:**")
                    for category, features in st.session_state.region_data.items():
                        if features:  # 해당 카테고리에 데이터가 있는 경우
                            st.write(f"- **{category}**: {len(features)}개 지표")
        
        # 기존 분석 결과 이미지들 표시 (오류 처리 강화)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### [1. 데이터 스케일링]")
            img = load_image_safely(img_path4)
            if img:
                st.image(img, caption="(1) 데이터 스케일링 확인", width=600, clamp=False)
            
            st.markdown("""
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
            img = load_image_safely(img_path3)
            if img:
                st.image(img, caption="(3) 시도별 평균 빈집지수, 사용된 방법론 분포 등", width=600, clamp=False)

            img = load_image_safely(img_path6)
            if img:
                st.image(img, caption="(3) 방법론별 평균 빈집지수", width=600, clamp=False)

            st.markdown("#### [4. 변수 중요도(Feature) - 빈집비율_합계]")
            img = load_image_safely(img_path9)
            if img:
                st.image(img, caption="(4) 빈집비율_합계 - Shap Summary Plot", width=600, clamp=False)

            img = load_image_safely(img_path10)
            if img:
                st.image(img, caption="(4) 빈집비율_합계 - Shap Waterfall Plot (샘플링)", width=600, clamp=False)

            st.markdown("#### [5. 빈집 예측 ML 모델 비교]")
            img = load_image_safely(img_path20)
            if img:
                st.image(img, caption="(5) 전체 ML 모델 성능 비교", width=600, clamp=False)

        with col2:
            st.markdown("#### [2. EDA (탐색적 데이터 분석)]")
            img = load_image_safely(img_path1)
            if img:
                st.image(img, caption="(2) 2023년 시도별 평균 빈집비율", width=600, clamp=False)
            
            img = load_image_safely(img_path2)
            if img:
                st.image(img, caption="(2) 2023년 시도별 평균 빈집비율_단독주택", width=600, clamp=False)

            st.markdown("#### [3. 요인분석, PCA 등 빈집지수 생성]")
            img = load_image_safely(img_path7)
            if img:
                st.image(img, caption="(3) 시도별 빈집비율 가중치 비교", width=600, clamp=False)

            img = load_image_safely(img_path8)
            if img:
                st.image(img, caption="(3) 2019년 경상북도 요인별 분산 설명력", width=600, clamp=False)

            st.markdown("#### [4. 변수 중요도(Feature) - 빈집비율_합계]")
            img = load_image_safely(img_path11)
            if img:
                st.image(img, caption="(4) 빈집비율_연립주택 - Shap Summary Plot", width=600, clamp=False)

            img = load_image_safely(img_path12)
            if img:
                st.image(img, caption="(4) 빈집비율_연립주택 - Shap Waterfall Plot (샘플링)", width=600, clamp=False)

# 챗봇 페이지로 이동 버튼
chatbot_clicked = False
if st.button("빈집 솔루션 챗봇"):
    st.session_state.type_of_case = "Chatbot"
    switch_page("chatbot")

# =============================================================================
# 앱 종료 시 리소스 정리
# =============================================================================
import atexit

def cleanup_on_exit():
    """앱 종료 시 리소스 정리"""
    try:
        # 필요 시 추가 정리 작업 수행
        pass
    except Exception:
        pass

# 종료 시 정리 함수 등록
atexit.register(cleanup_on_exit)
