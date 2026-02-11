import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. 頁面配置
st.set_page_config(page_title="金融信用預測儀表板", layout="wide")

# 2. 定義快取函式 (提升效能)
@st.cache_resource
def load_model(model_name: str):
    model_files = {
        "KNN": "k-nearest_neighbors_pipeline.joblib",
        "LogisticRegression": "logistic_regression_pipeline.joblib",
        # "RandomForest": "randomforest_classifier_pipeline.joblib",  # 你目前沒用就先別開
        "XGBoost": "xgboost_classifier_pipeline.joblib",
    }

    if model_name not in model_files:
        st.error(f"找不到模型設定：{model_name}")
        st.stop()

    path = model_files[model_name]
    if not os.path.exists(path):
        st.error(f"找不到模型檔：{path}\n\n請確認已上傳到 GitHub repo 根目錄，且檔名完全一致。")
        st.stop()

    return joblib.load(path)

@st.cache_data
def load_data():
    local_csv = "UCI_Credit_Card.csv"

    # ✅ Debug：讓你在雲端直接看得到「到底有哪些檔案」
    # 部署成功後若你不想顯示，可以把下面兩行註解掉
    # st.write("Files in repo:", os.listdir("."))
    # st.write("CSV exists?", os.path.exists(local_csv))

    if not os.path.exists(local_csv):
        st.error(
            f"找不到資料檔：{local_csv}\n\n"
            "請確認：\n"
            "1) 檔案已上傳到 GitHub repo 根目錄\n"
            "2) 檔名大小寫完全一致（UCI_Credit_Card.csv）\n"
            "3) 不是 UCI_Credit_Card (1).csv 或 csv.csv"
        )
        st.stop()

    df = pd.read_csv(local_csv)

    # 分離特徵與標籤 (為了之後預測用)
    cols = df.columns.tolist()
    possible_labels = [
        "default payment next month",
        "default.payment.next.month",
        "default_payment_next_month",
        "default.payment_next_month",
    ]

    label_col = next((c for c in cols if c in possible_labels), None)
    if label_col is None:
        for c in cols:
            if "default" in c.lower() and "next" in c.lower():
                label_col = c
                break

    if label_col is None:
        st.error("找不到標籤欄位 (default ...)，請檢查 CSV 欄位名稱")
        st.stop()

    id_col = next((c for c in cols if c.lower() == "id"), None)
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

# ✅ 先拿掉 RandomForest（你目前 model_files 沒放，選到就會炸）
selected_name = st.sidebar.selectbox(
    "請選擇分類模型：",
    ["KNN", "LogisticRegression", "XGBoost"],
)

model = load_model(selected_name)

st.sidebar.divider()
st.sidebar.info(
    f"當前模型：{selected_name}\n\n"
    "這是一個包含 Scaler / PCA / Classifier 的完整 Pipeline。"
)

# --- 右側主畫面 ---
st.title("💳 信用卡違約風險預測展示")

# A. 數據概覽
st.subheader("📋 數據集概覽 (前 10 筆樣本)")
st.dataframe(df_full.head(10), use_container_width=True)

st.divider()

# B. 隨機預測區塊
st.subheader("🎯 即時預測測試")

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = None

if st.button("🎲 隨機抽取一個樣本進行預測"):
    st.session_state.sample_idx = np.random.randint(0, len(X))

if st.session_state.sample_idx is not None:
    idx = st.session_state.sample_idx

    sample_data = X.iloc[[idx]]
    actual_label = y.iloc[idx]

    st.write(f"**抽取的樣本索引：** `{idx}`")
    st.dataframe(sample_data, use_container_width=True)

    prediction = model.predict(sample_data)[0]

    # 有些 pipeline / 模型可能沒有 predict_proba，保護一下
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(sample_data)[0][1]

    st.subheader("🚀 預測結果")

    col1, col2, col3 = st.columns(3)

    with col1:
        res_text = "⚠️ 違約" if prediction == 1 else "✅ 正常"
        st.metric("模型預測", res_text)

    with col2:
        actual_text = "⚠️ 違約" if actual_label == 1 else "✅ 正常"
        st.metric("真實情況", actual_text)

    with col3:
        if prob is not None:
            st.metric("違約機率", f"{prob:.2%}")
        else:
            st.metric("違約機率", "此模型不支援")

    if prediction == actual_label:
        st.success("🎉 預測正確！該模型成功捕捉到樣本特徵。")
    else:
        st.error("❌ 預測失誤。這反映了模型在邊際樣本上的侷限性。")
