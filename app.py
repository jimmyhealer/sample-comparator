import streamlit as st
import pandas as pd
import json
import os
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import base64
import matplotlib.font_manager as fm
import platform

# 設置中文字體支持
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'Noto Sans CJK TC']

plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# Page configuration
st.set_page_config(
    page_title="資料集橫向比對分析工具",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for storing data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'comments' not in st.session_state:
    st.session_state.comments = {}
if 'tags' not in st.session_state:
    st.session_state.tags = {}
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = None
if 'visible_strategies' not in st.session_state:
    st.session_state.visible_strategies = []
if 'strategy_order' not in st.session_state:
    st.session_state.strategy_order = []

# Title
st.title("資料集橫向比對分析工具")
st.markdown("協助使用者以「樣本為核心」橫向比對不同策略或模型的結果")

# Function to create a download link
def get_download_link(data, filename, filetype):
    if filetype == 'json':
        data_str = json.dumps(data, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(data_str.encode('utf-8')).decode()
        href = f'data:application/json;charset=utf-8;base64,{b64}'
    elif filetype == 'csv':
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode('utf-8')).decode()
        href = f'data:text/csv;charset=utf-8;base64,{b64}'
    
    download_link = f'<a href="{href}" download="{filename}">下載 {filename}</a>'
    return download_link

# Main function to process uploaded files
def process_uploaded_files(uploaded_files):
    all_data = {}
    strategy_names = []
    
    # Read and process each file
    for uploaded_file in uploaded_files:
        try:
            # Extract strategy name from filename without extension
            strategy_name = os.path.splitext(uploaded_file.name)[0]
            strategy_names.append(strategy_name)
            
            # Parse JSON file
            content = uploaded_file.getvalue().decode("utf-8")
            data = json.loads(content)
            
            # Store data by sentence to merge later
            for item in data:
                sentence = item.get("sentence", "")
                if not sentence:
                    continue
                
                if sentence not in all_data:
                    all_data[sentence] = {
                        "sentence": sentence,
                        "label": item.get("label", ""),
                        "strategies": {}
                    }
                
                # Add strategy result to the sample
                all_data[sentence]["strategies"][strategy_name] = {
                    "input": item.get("input", ""),
                    "output": item.get("output", {}),
                    "is_correct": (
                        item.get("output", {}).get("answer", "") == 
                        item.get("label", "")
                    )
                }
        except Exception as e:
            st.error(f"處理檔案 {uploaded_file.name} 時發生錯誤: {str(e)}")
            return None
    
    # Convert to list and create DataFrame
    processed_data = {
        "samples": list(all_data.values()),
        "strategy_names": strategy_names
    }
    
    return processed_data

# Function to create a DataFrame for the main table view
def create_dataframe(processed_data):
    if not processed_data:
        return None
    
    samples = processed_data["samples"]
    strategy_names = processed_data["strategy_names"]
    
    # Create basic DataFrame
    df_rows = []
    for sample in samples:
        row = {
            "sentence": sample["sentence"],
            "label": sample["label"],
        }
        
        # Add strategy results
        for strategy in strategy_names:
            strategy_data = sample["strategies"].get(strategy, {})
            row[f"{strategy}_answer"] = strategy_data.get("output", {}).get("answer", "")
            row[f"{strategy}_correct"] = strategy_data.get("is_correct", False)
        
        # Count correct/incorrect answers
        correct_count = sum(1 for s in strategy_names if sample["strategies"].get(s, {}).get("is_correct", False))
        row["correct_count"] = correct_count
        row["incorrect_count"] = len(strategy_names) - correct_count
        
        df_rows.append(row)
    
    return pd.DataFrame(df_rows)

# Sidebar for file upload and controls
with st.sidebar:
    st.header("上傳檔案")
    uploaded_files = st.file_uploader(
        "上傳JSON檔案 (每個對應一種策略結果)",
        type=["json"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"已上傳 {len(uploaded_files)} 個檔案")
        
        # Process uploaded files
        if st.session_state.data is None:
            with st.spinner("處理上傳的檔案..."):
                processed_data = process_uploaded_files(uploaded_files)
                if processed_data:
                    st.session_state.data = processed_data
                    st.session_state.df = create_dataframe(processed_data)
                    
                    # Initialize visible strategies and order
                    if not st.session_state.visible_strategies:
                        st.session_state.visible_strategies = processed_data["strategy_names"].copy()
                    if not st.session_state.strategy_order:
                        st.session_state.strategy_order = list(range(len(processed_data["strategy_names"])))
                    
                    st.success("檔案處理完成!")

        # 策略顯示設定 - 移到側邊欄
        if hasattr(st.session_state, 'df') and st.session_state.df is not None:
            all_strategy_names = st.session_state.data["strategy_names"]
            
            st.header("策略設定")
            
            # 顯示策略設定
            with st.expander("選擇要顯示的策略", expanded=True):                
                # 選擇要顯示的策略
                visible_strategies = []
                for strategy in all_strategy_names:
                    is_visible = st.checkbox(strategy, value=strategy in st.session_state.visible_strategies, key=f"vis_{strategy}")
                    if is_visible:
                        visible_strategies.append(strategy)
                st.session_state.visible_strategies = visible_strategies
        
        # Export section
        st.header("匯出結果")
        export_format = st.radio("選擇匯出格式:", ["JSON", "CSV"])
        if st.button("匯出資料"):
            if st.session_state.data is not None:
                if export_format == "JSON":
                    # Prepare data for JSON export with comments and tags
                    export_data = []
                    for sample in st.session_state.data["samples"]:
                        sample_dict = {
                            "sentence": sample["sentence"],
                            "label": sample["label"],
                            "strategies": {}
                        }
                        
                        # Add comments and tags if they exist
                        sample_key = sample["sentence"]
                        if sample_key in st.session_state.comments:
                            sample_dict["comment"] = st.session_state.comments[sample_key]
                        if sample_key in st.session_state.tags:
                            sample_dict["tags"] = st.session_state.tags[sample_key]
                        
                        # Add strategy data
                        for strategy_name, strategy_data in sample["strategies"].items():
                            sample_dict["strategies"][strategy_name] = {
                                "input": strategy_data["input"],
                                "output": strategy_data["output"],
                                "is_correct": strategy_data["is_correct"]
                            }
                        
                        export_data.append(sample_dict)
                    
                    # Create download link for JSON
                    st.markdown(
                        get_download_link(export_data, "llm_analysis_results.json", "json"),
                        unsafe_allow_html=True
                    )
                else:  # CSV
                    # Prepare data for CSV export
                    if hasattr(st.session_state, 'df'):
                        export_df = st.session_state.df.copy()
                        
                        # Add comments and tags to DataFrame
                        export_df["comment"] = export_df["sentence"].apply(
                            lambda s: st.session_state.comments.get(s, "")
                        )
                        export_df["tags"] = export_df["sentence"].apply(
                            lambda s: ", ".join(st.session_state.tags.get(s, []))
                        )
                        
                        # Create download link for CSV
                        st.markdown(
                            get_download_link(export_df, "llm_analysis_results.csv", "csv"),
                            unsafe_allow_html=True
                        )

# Display main results table if data exists
if hasattr(st.session_state, 'df') and st.session_state.df is not None:
    df = st.session_state.df
    all_strategy_names = st.session_state.data["strategy_names"]
    
    # 按照用戶設定的順序對可見策略進行排序
    strategy_names = []
    for strategy in all_strategy_names:
        if strategy in st.session_state.visible_strategies:
            strategy_names.append(strategy)
    strategy_names = sorted(strategy_names, key=lambda s: st.session_state.strategy_order[all_strategy_names.index(s)])
    
    # Add filters
    st.header("比對結果表格")
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by strategy correct/incorrect
        filter_options = ["全部"] + strategy_names
        filter_strategy = st.selectbox("按策略篩選:", filter_options)
    
    with col2:
        # Filter by correct/incorrect
        filter_status = st.selectbox("按狀態篩選:", ["全部", "正確", "錯誤"])
    
    # Apply filters
    filtered_df = df.copy()
    if filter_strategy != "全部":
        if filter_status == "正確":
            filtered_df = filtered_df[filtered_df[f"{filter_strategy}_correct"] == True]
        elif filter_status == "錯誤":
            filtered_df = filtered_df[filtered_df[f"{filter_strategy}_correct"] == False]
    
    # Sort options
    sort_options = ["錯誤數量 (多->少)", "錯誤數量 (少->多)"]
    sort_by = st.selectbox("排序方式:", sort_options)
    
    if sort_by == "錯誤數量 (多->少)":
        filtered_df = filtered_df.sort_values("incorrect_count", ascending=False)
    elif sort_by == "錯誤數量 (少->多)":
        filtered_df = filtered_df.sort_values("incorrect_count", ascending=True)
    
    # Display table with colored cells based on correctness
    display_df = filtered_df[["sentence", "label", "correct_count", "incorrect_count"]].copy()
    
    # Add strategy answers with colored backgrounds
    for strategy in strategy_names:
        def highlight_answer(answer, is_correct):
            color = "background-color: #c6ebc9" if is_correct else "background-color: #ffa8a8"
            return color
        
        display_df[strategy] = filtered_df.apply(
            lambda row: f"{row[f'{strategy}_answer']}",
            axis=1
        )
    
    # Function to apply color styling based on correctness
    def color_correct_incorrect(row):
        styles = [""] * len(row)
        for i, col in enumerate(row.index):
            if col in strategy_names:
                strategy = col
                is_correct = filtered_df.loc[row.name, f"{strategy}_correct"] if f"{strategy}_correct" in filtered_df.columns else False
                styles[i] = highlight_answer("", is_correct)
        return styles
    
    # Apply styling
    styled_df = display_df.style.apply(color_correct_incorrect, axis=1)
    
    # Display table
    st.dataframe(styled_df, use_container_width=True)
    
    selected_index = st.selectbox(
        "樣本索引",
        options=list(range(len(filtered_df))),
        format_func=lambda i: f"{i}. {df.iloc[i]['sentence']}"
    )
    
    if selected_index is not None:
        st.session_state.selected_sample = df.iloc[selected_index]["sentence"]
    
    # Display detailed view
    if st.session_state.selected_sample:
        selected_sample = next(
            (s for s in st.session_state.data["samples"] 
             if s["sentence"] == st.session_state.selected_sample),
            None
        )
        
        if selected_sample:
            st.markdown(f"**正確答案 (Label):** {selected_sample['label']}")
            
            # Comments section
            sample_key = selected_sample["sentence"]
            
            # Display strategy details in tabs - using visible strategies in specified order
            tabs = st.tabs(strategy_names)
            for i, strategy in enumerate(strategy_names):
                with tabs[i]:
                    strategy_data = selected_sample["strategies"].get(strategy, {})
                    is_correct = strategy_data.get("is_correct", False)
                    st.markdown(f"**結果:** {'✅ 正確' if is_correct else '❌ 錯誤'}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Input (Prompt)")
                        st.text_area(
                            "Input",
                            value=strategy_data.get("input", ""),
                            height=200,
                            key=f"input_{strategy}_{sample_key}",
                            disabled=True
                        )
                    
                    with col2:
                        st.subheader("Output")
                        output = strategy_data.get("output", {})
                        st.markdown(f"**Answer:** {output.get('answer', '')}")
                        
                        if "reasoning" in output:
                            st.subheader("Reasoning")
                            st.text_area(
                                "Reasoning",
                                value=output.get("reasoning", ""),
                                height=150,
                                key=f"reasoning_{strategy}_{sample_key}",
                                disabled=True
                            )
                        
                        if "response" in output:
                            st.subheader("Response")
                            st.text_area(
                                "Response",
                                value=output.get("response", ""),
                                height=150,
                                key=f"response_{strategy}_{sample_key}",
                                disabled=True
                            )
    
            if sample_key not in st.session_state.comments:
                st.session_state.comments[sample_key] = ""
            
            st.session_state.comments[sample_key] = st.text_area(
                "評論:",
                value=st.session_state.comments[sample_key],
                key=f"comment_{sample_key}"
            )
            
            # Tags section
            if sample_key not in st.session_state.tags:
                st.session_state.tags[sample_key] = []
            
            # Tag input and addition
            new_tag = st.text_input("新增標籤:", key=f"new_tag_{sample_key}")
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("新增標籤"):
                    if new_tag and new_tag not in st.session_state.tags[sample_key]:
                        st.session_state.tags[sample_key].append(new_tag)
            
            # Show current tags
            with col2:
                st.write("目前標籤:", ", ".join(st.session_state.tags[sample_key]) if st.session_state.tags[sample_key] else "無")

    # Extended features
    st.header("策略交集分析")
    if len(strategy_names) > 1:
        # Create a table showing overlap of errors
        st.markdown("### 錯誤重疊分析")
        overlap_data = []
        
        for s1, s2 in combinations(strategy_names, 2):
            # Count samples where both strategies are incorrect
            both_incorrect = filtered_df[
                (filtered_df[f"{s1}_correct"] == False) & 
                (filtered_df[f"{s2}_correct"] == False)
            ].shape[0]
            
            # Count samples where only one strategy is incorrect
            s1_only_incorrect = filtered_df[
                (filtered_df[f"{s1}_correct"] == False) & 
                (filtered_df[f"{s2}_correct"] == True)
            ].shape[0]
            
            s2_only_incorrect = filtered_df[
                (filtered_df[f"{s1}_correct"] == True) & 
                (filtered_df[f"{s2}_correct"] == False)
            ].shape[0]
            
            # Count total errors for each strategy
            s1_total_incorrect = filtered_df[filtered_df[f"{s1}_correct"] == False].shape[0]
            s2_total_incorrect = filtered_df[filtered_df[f"{s2}_correct"] == False].shape[0]
            
            # Calculate overlap percentage
            overlap_percentage = 0
            if s1_total_incorrect > 0 and s2_total_incorrect > 0:
                overlap_percentage = (both_incorrect / min(s1_total_incorrect, s2_total_incorrect)) * 100
            
            overlap_data.append({
                "策略對": f"{s1} vs {s2}",
                "共同錯誤數": both_incorrect,
                f"{s1}獨有錯誤": s1_only_incorrect,
                f"{s2}獨有錯誤": s2_only_incorrect,
                "重疊率": f"{overlap_percentage:.1f}%"
            })
        
        overlap_df = pd.DataFrame(overlap_data)
        st.dataframe(overlap_df, use_container_width=True)
        
        # Compare overall performance
        st.markdown("### 策略總體表現比較")
        performance_data = []
        
        for strategy in strategy_names:
            correct = filtered_df[filtered_df[f"{strategy}_correct"] == True].shape[0]
            incorrect = filtered_df[filtered_df[f"{strategy}_correct"] == False].shape[0]
            accuracy = correct / (correct + incorrect) * 100 if (correct + incorrect) > 0 else 0
            
            performance_data.append({
                "策略": strategy,
                "正確數": correct,
                "錯誤數": incorrect,
                "準確率": f"{accuracy:.2f}%"
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values("準確率", ascending=False)
        st.dataframe(performance_df, use_container_width=True)

# Initial instructions if no data
if not uploaded_files:
    st.info("請上傳JSON檔案以開始分析")
    
    # Example JSON format
    st.markdown("### 輸入JSON格式範例:")
    example = {
        "input": "...",
        "label": "gao1",
        "sentence": "相同句子 (作為 ID 使用，所以比較的對象要一樣)",
        "output": {
            "answer": "gao1",
            "reasoning": "...",
            "response": "..."
        }
    }
    st.code(json.dumps(example, indent=2, ensure_ascii=False), language="json") 