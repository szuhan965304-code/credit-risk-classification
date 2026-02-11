import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. 頁面配置
st.set_page_config(page_title="金融信用預測儀表板", layout="wide")

# 2. 定義快取函式 (提升效能)
@st.cache_resource
def load_model(model_name):
    # 這裡的檔名需與你下載的 joblib 檔案名稱一致
    model_files = {
        "KNN": "k-nearest_neighbors_pipeline.joblib",
        "LogisticRegression": "logistic_regression_pipeline.joblib",
        #"RandomForest": "randomforest_classifier_pipeline.joblib",
        "XGBoost": "xgboost_classifier_pipeline.joblib"
    }
    return joblib.load(model_files[model_name])

@st.cache_data
def load_data():
    import os
    local_csv = "UCI_Credit_Card.csv"
    if os.path.exists(local_csv):
        df = pd.read_csv(local_csv)
    else:
        url = "https://raw.githubusercontent.com/ywang166/Credit-Card-Default-Prediction/master/data/default%20of%20credit%20card%20clients.csv"
        df = pd.read_csv(url, skiprows=1)

    # 分離特徵與標籤 (為了之後預測用)
    cols = df.columns.tolist()
    possible_labels = [
        'default payment next month',
        'default.payment.next.month',
        'default_payment_next_month',
        'default.payment_next_month'
    ]
    label_col = next((c for c in cols if c in possible_labels), None)
    if label_col is None:
        for c in cols:
            if 'default' in c.lower() and 'next' in c.lower():
                label_col = c
                break
    if label_col is None:
        raise ValueError("找不到標籤欄位 (default ...)，請檢查 CSV 欄位名稱")

    id_col = next((c for c in cols if c.lower() == 'id'), None)
    drop_cols = [label_col]
    if id_col:
        drop_cols.insert(0, id_col)

    X = df.drop(drop_cols, axis=1)
    y = df[label_col]
    return df, X, y

# 3. 載入資料
df_full, X, y = load_data()

# --- 左側選單 (Sidebar) ---
st.sidebar.title("🤖 模型控制中心")
selected_name = st.sidebar.selectbox(
    "請選擇分類模型：",
    ["KNN", "LogisticRegression", "RandomForest", "XGBoost"]
)
model = load_model(selected_name)

st.sidebar.divider()
st.sidebar.info(f"當前模型：{selected_name}\n\n這是一個包含 Scaler, PCA 與 Classifier 的完整 Pipeline。")

# --- 右側主畫面 ---
st.title("💳 信用卡違約風險預測展示")

# A. 數據概覽
st.subheader("📋 數據集概覽 (前 10 筆樣本)")
st.dataframe(df_full.head(10), use_container_width=True)

st.divider()

# B. 隨機預測區塊
st.subheader("🎯 即時預測測試")

# 初始化 session_state 用於儲存抽樣結果
if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = None

if st.button("🎲 隨機抽取一個樣本進行預測"):
    st.session_state.sample_idx = np.random.randint(0, len(X))

# 如果已經抽樣，則進行顯示與預測
if st.session_state.sample_idx is not None:
    idx = st.session_state.sample_idx
    
    # 取出單筆資料 (DataFrame 格式，Pipeline 才能吃)
    sample_data = X.iloc[[idx]]
    actual_label = y.iloc[idx]
    
    st.write(f"**抽取的樣本索引：** `{idx}`")
    st.dataframe(sample_data)
    
    # 執行 Pipeline 預測 (自動內含 Scaling 與 PCA)
    prediction = model.predict(sample_data)[0]
    # 預測機率 (XGB, RF, LR 支援，KNN 也支援)
    prob = model.predict_proba(sample_data)[0][1]
    
    # --- 下方顯示結果 ---
    st.subheader("🚀 預測結果")
    
    # 使用欄位排版顯示指標
    col1, col2, col3 = st.columns(3)
    
    with col1:
        res_text = "⚠️ 違約" if prediction == 1 else "✅ 正常"
        st.metric("模型預測", res_text)
        
    with col2:
        actual_text = "⚠️ 違約" if actual_label == 1 else "✅ 正常"
        st.metric("真實情況", actual_text)
        
    with col3:
        st.metric("違約機率", f"{prob:.2%}")

    # 比對結果
    if prediction == actual_label:
        st.success("🎉 預測正確！該模型成功捕捉到樣本特徵。")
    else:

        st.error("❌ 預測失誤。這反映了模型在邊際樣本上的侷限性。")
