---
title: "How To Cluster Joint Sets: A K-Means Streamlit Approach"
excerpt: "Exemple <br/><img src='/images/k_means_stereonet_report.png'>"
collection: portfolio
---

This is a mini-project for the Introduction to Python course, titled: How To Cluster Joint Sets: A K-Means Streamlit Approach.

This is a Python application designed to automatically classify joint sets using the K-Means clustering algorithm, an Unsupervised Machine Learning method for grouping data.

The application is built to divide the orientation data into K groups (Joint Sets) and plot the results on a Stereonet, featuring the following capabilities:

### Key Features

Platform: Runs as a web application (using the Streamlit library).

Data Input: Accepts .csv or .xlsx files containing Dip Direction and Dip Angle values.

Clustering Control: Allows the user to select the number of groups (the K-value).

Plot Type Selection: Users can choose between plotting the data as Poles or Planes.

Customization: Supports changing the plot's Font style.

Export 1 (Visualization): Enables downloading the generated Stereonet plot image.

Export 2 (Data): Allows exporting the original data with the new Joint Set labels added as a .csv file.

### Objective

The primary objective is to aid in the grouping of joint or fracture data through computational analysis. This approach prevents errors arising from manual polygon picking and significantly reduces processing time. The resulting classified joint set file (.csv) can then be seamlessly used as input for subsequent stereonet plotting or analysis applications.

### Download
- [Download Python script](/files/kmean4.py)
- [Download Data Example 1](/files/joint.csv)
- [Download Data Example 2](/files/joint2.csv)

### Python code
```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import io

def orientation_to_vector(dd, dip):
    """Converts dip direction (dd) and dip angle (dip) to a 3D vector (nx, ny, nz)"""
    dd_rad = np.radians(dd)
    dip_rad = np.radians(dip)

    # Use Right-Hand Rule (RHR) for strike: strike = dd - 90
    strike = (dd - 90) % 360
    strike_rad = np.radians(strike)

    # Calculate vector components (Pole vector calculation)
    nx = np.sin(dip_rad) * np.sin(strike_rad)
    ny = np.sin(dip_rad) * np.cos(strike_rad)
    nz = np.cos(dip_rad)

    return [nx, ny, nz]

@st.cache_data
def load_data(uploaded_file):
    """Loads and validates data from an uploaded CSV or XLSX file."""
    # Read data based on file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("ไฟล์ต้องเป็น CSV หรือ XLSX")
        return None

    df.columns = df.columns.str.lower().str.strip()
    required_cols = ["dip direction", "dip angle"]
    
    if not all(col in df.columns for col in required_cols):
        st.error("❌ ไฟล์ต้องมีคอลัมน์: **dip direction** และ **dip angle** (ตัวพิมพ์เล็ก-ใหญ่ได้)")
        return None

    df = df[required_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if df.empty:
         st.error("❌ ข้อมูลที่จำเป็น (dip direction, dip angle) เป็นค่าว่างหรือไม่ใช่ตัวเลข")
         return None

    return df

def plot_stereonet(df, k_value, plot_type, font_name):
    """Performs K-Means clustering and plots the results on a stereonet."""
    st.subheader(f"📊 ผลลัพธ์การจัดกลุ่ม K={k_value} ({plot_type.capitalize()} Plot)")

    vectors = np.array([orientation_to_vector(dd, d)
                        for dd, d in zip(df["dip direction"], df["dip angle"])])

    try:
        model = KMeans(n_clusters=k_value, random_state=42, n_init='auto')
        labels = model.fit_predict(vectors)
        df["joint_set"] = labels
    except ValueError as e:
        st.error(f"❌ Error ในการทำ K-Means: ตรวจสอบจำนวนข้อมูล ({len(df)}) และ K value")
        st.exception(e)
        return None
    
    plt.rcParams["font.family"] = font_name

    fig_report = plt.figure(figsize=(7, 7)) 
    ax = fig_report.add_subplot(111, projection='stereonet')
    
    colors = plt.cm.get_cmap("Set1", k_value)

    for i in range(k_value):
        subset = df[df["joint_set"] == i]
        dd_subset = subset["dip direction"].values
        dip_subset = subset["dip angle"].values

        if plot_type == "Plane":
             ax.plane(dd_subset, dip_subset,
                      color=colors(i), alpha=0.5, 
                      label=f"Set {i} (n={len(subset)})")
        
        elif plot_type == "Pole":
            ax.pole(dd_subset, dip_subset,
                    marker="o", markersize=5, color=colors(i), alpha=0.7, label=f"Set {i} (n={len(subset)})")
        
    ax.legend(loc="lower left", title="Joint Sets")
    
    st.pyplot(fig_report)
    
    st.success("✅ กราฟถูกแสดงผลและจัดกลุ่มเสร็จสิ้น!")
    
    buf = io.BytesIO()
    fig_report.savefig(buf, format="png", dpi=300)
    st.download_button(
        label="🖼️ ดาวน์โหลดภาพ Stereonet สำหรับรายงาน (.png)",
        data=buf.getvalue(),
        file_name='k-means_stereonet_report.png',
        mime='image/png'
    )
    
    st.subheader("ข้อมูลพร้อม Joint Set Label")
    st.dataframe(df)

    @st.cache_data
    def convert_df_to_csv(df):
        # To CSV without index
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df)

    st.markdown("---")
    st.subheader("⬇️ ดาวน์โหลดข้อมูล Joint Set")
    st.download_button(
        label="คลิกเพื่อดาวน์โหลดข้อมูลพร้อม Joint Set Label (.csv)",
        data=csv_data,
        file_name='k-means_joint_sets.csv',
        mime='text/csv',
    )
    
    return df

def main_app():
    st.title("🪨 K-Means Stereonet Plotter (Web App)")
    st.markdown("""
        แอปพลิเคชันนี้ใช้ **K-Means Clustering** เพื่อจัดกลุ่มชุดข้อมูลทิศทาง (Dip Direction & Dip Angle)
        และแสดงผลบน **Stereonet**
    """)
    st.markdown("---")

    
    uploaded_file = st.file_uploader(
        "1. อัปโหลดไฟล์ข้อมูล (.CSV หรือ .XLSX)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ โหลดข้อมูล {len(df)} แถว สำเร็จแล้ว")
            st.dataframe(df.head())
            st.markdown("---")

            st.sidebar.header("⚙️ การตั้งค่าการ Plot")

            k_value = st.sidebar.slider(
                "2. จำนวนคลัสเตอร์ K (Joint Sets):",
                min_value=2, max_value=10, value=3, step=1,
                help="จำนวนกลุ่มที่ K-Means จะจัด (K=2 ถึง K=10)"
            )

            plot_type = st.sidebar.radio(
                "3. เลือก Plot Type:",
                ("Pole", "Plane"),
                help="Pole: แสดงจุด (เหมาะสำหรับ Clustering), Plane: แสดงเส้นระนาบ"
            )

            st.sidebar.markdown("---")
            st.sidebar.subheader("ตัวเลือกกราฟขั้นสูง")

            font_name = st.sidebar.selectbox(
                "4. เลือก Font สำหรับกราฟ:",
                options=["Tahoma", "Arial", "Times New Roman", "DejaVu Sans"],
                index=0, 
                help="Font ที่ใช้ใน Title, Label, และ Legend"
            )
            
            if st.button("🚀 ประมวลผลและแสดง Stereonet"):
                # Run the clustering and plotting with the remaining parameters
                plot_stereonet(df, k_value, plot_type, font_name)
        else:
            st.warning("กรุณาตรวจสอบโครงสร้างไฟล์และลองใหม่อีกครั้ง")


if __name__ == "__main__":
    main_app()
```
### Streamlit run
<img src='/images/web1.png'>
<img src='/images/web2.png'>
<img src='/images/web3.png'>
