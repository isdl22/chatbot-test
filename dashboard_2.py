import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
def load_data():
    data = pd.read_excel('1대1 컨설팅 데이터.xlsx')  # 모든 시트를 불러옴
    return data

def main():
    st.title("고객 컨설팅 데이터")

    # 데이터 불러오기
    data = load_data()

    st.sidebar.header("Navigation")
    selected_section = st.sidebar.radio("Select a Section", ["기본 정보", "경력 기술서 정보", "설문 정보", "고민 사항"])

   # 고객 이름 선택
    name_list = data['성함'].unique()
    name_input = st.selectbox("고객 이름을 선택하세요:", name_list)


    # Generate charts only when a name is entered
    if name_input:
        customer_data = data[data['성함'] == name_input]
        basic_info_columns = ['직무', '대학교', '전공', '학점', '나이','피드백 받고 싶은 영역']
        feedback_columns = ['피드백 받고 싶은 내용에 대해 조금 세부적으로 본인의 상황을 설명해주세요.']
        project_columns = ['프로젝트명/업무명','프로젝트 기간','프로젝트 성과','프로젝트 역할','프로젝트 기술']
        activity_columns = ['활동 기간',	'활동 내용',	'수상 기간',	'수상 내역',	'어학/자격증']
        unique_customer_data = customer_data[basic_info_columns + feedback_columns].drop_duplicates().reset_index(drop=True)

        if selected_section == "기본 정보":
            st.subheader("기본 이력정보")
            if not unique_customer_data.empty:
            # 기본 정보 표시
                st.subheader("기본 정보")
                st.table(unique_customer_data[basic_info_columns].T)
                st.subheader("피드백 정보")
                st.table(unique_customer_data[feedback_columns])
                st.subheader("활동 정보")
                activity_info = customer_data[activity_columns].dropna()
                if not activity_info.empty:
                    st.table(activity_info)    
        elif selected_section == "경력 기술서 정보":
            st.subheader("경력 기술서 정보")
            project_info = customer_data[project_columns].dropna()
            if not project_info.empty:
                st.table(project_info)
            # Add the radar chart here
        elif selected_section == "설문 정보":
            st.subheader("설문 정보")
            if name_input in data['성함'].values:
                labels = ["현회사_만족도", "현회사_재선택_여부", "23년_나의_성과_평가", "상사_평가", "상사_평가_만족도"]
                individual_columns = ["현회사_만족도", "현회사_재선택_여부", "23년_나의_성과_평가", "상사_평가", "상사_평가_만족도"]
                average_columns = ["AVG_현회사_만족도", "AVG_현회사_재선택_여부", "AVG_23년_나의_성과_평가", "AVG_상사_평가", "AVG_상사_평가_만족도"]
                    # Create and display the radar chart
                fig = create_radar_chart_with_individual_comparison(data, name_input, individual_columns, average_columns, labels)
                st.pyplot(fig)
        elif selected_section == "고민 사항":
            st.subheader("고민 사항")
            st.table(customer_data["고민"].drop_duplicates())
        else:
            st.write("Name not found in the data.")


def create_radar_chart_with_individual_comparison(data, individual_name, individual_columns, average_columns, labels):
    # Font settings
    rcParams['font.family'] = 'NanumBarunGothic'

    # Filter data for the specified individual
    individual_data = data[data['성함'] == individual_name][individual_columns]

    # Calculate distinct average values for the individual's responses
    individual_avg_values = individual_data.mean().values

    # Extract the overall average data
    overall_average_data = data.loc[0, average_columns].values

    # Prepare the data for the radar chart
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    individual_avg_values = np.concatenate((individual_avg_values, [individual_avg_values[0]]))
    overall_average_data = np.concatenate((overall_average_data, [overall_average_data[0]]))
    angles += angles[:1]

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, individual_avg_values, color='blue', alpha=0.25)
    ax.plot(angles, individual_avg_values, color='blue', linewidth=2, label=f'Individual - {individual_name}')
    ax.fill(angles, overall_average_data, color='green', alpha=0.25)
    ax.plot(angles, overall_average_data, color='green', linewidth=2, label='Overall Average')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    return fig
    # Add other sections as needed

if __name__ == '__main__':
    main()
