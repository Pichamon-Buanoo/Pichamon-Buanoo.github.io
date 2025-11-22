import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import io

# --------------------------------------------------
# Convert orientation to vector
# --------------------------------------------------
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

# --------------------------------------------------
# Read file and validate columns
# --------------------------------------------------
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

# --------------------------------------------------
# Plot Stereonet
# --------------------------------------------------
def plot_stereonet(df, k_value, plot_type, font_name):
    """Performs K-Means clustering and plots the results on a stereonet."""
    st.subheader(f"📊 ผลลัพธ์การจัดกลุ่ม K={k_value} ({plot_type.capitalize()} Plot)")

    # 1. Convert orientation to vectors
    vectors = np.array([orientation_to_vector(dd, d)
                        for dd, d in zip(df["dip direction"], df["dip angle"])])

    # 2. Run KMeans Clustering
    try:
        model = KMeans(n_clusters=k_value, random_state=42, n_init='auto')
        labels = model.fit_predict(vectors)
        df["joint_set"] = labels
    except ValueError as e:
        st.error(f"❌ Error ในการทำ K-Means: ตรวจสอบจำนวนข้อมูล ({len(df)}) และ K value")
        st.exception(e)
        return None

    # 3. Plotting Setup
    
    # กำหนด Font Global
    plt.rcParams["font.family"] = font_name
    
    # กำหนดขนาด Figure สำหรับรายงาน (ขนาดมาตรฐานสำหรับเอกสาร)
    fig_report = plt.figure(figsize=(7, 7)) 
    ax = fig_report.add_subplot(111, projection='stereonet')
    
    # Fixed: ใช้ ax.grid(True) เพื่อเปิด Grid (ค่า Default คือ 10 องศา) เท่านั้น
    ax.grid(True) 
    
    # ⚠️ ข้อความเตือนเรื่อง Grid Spacing ถูกลบออกแล้วตามคำขอ
    
    # Get distinct colors
    colors = plt.cm.get_cmap("Set1", k_value)

    # 4. Loop Plotting 
    for i in range(k_value):
        subset = df[df["joint_set"] == i]
        dd_subset = subset["dip direction"].values
        dip_subset = subset["dip angle"].values
        
        # Plot based on user choice (Pole or Plane)
        if plot_type == "Plane":
             # Plot great circles (planes) for all data points in the set
             ax.plane(dd_subset, dip_subset,
                      color=colors(i), alpha=0.5, 
                      label=f"Set {i} (n={len(subset)})")
        
        elif plot_type == "Pole":
            # Plot poles (points)
            ax.pole(dd_subset, dip_subset,
                    marker="o", markersize=5, color=colors(i), alpha=0.7, label=f"Set {i} (n={len(subset)})")
        
    ax.legend(loc="lower left", title="Joint Sets")
    
    # 5. แสดง Plot ใน Streamlit
    st.pyplot(fig_report)
    
    st.success("✅ กราฟถูกแสดงผลและจัดกลุ่มเสร็จสิ้น!")
    
    # 6. ปุ่มดาวน์โหลดรูปภาพ
    buf = io.BytesIO()
    fig_report.savefig(buf, format="png", dpi=300)
    st.download_button(
        label="🖼️ ดาวน์โหลดภาพ Stereonet สำหรับรายงาน (.png)",
        data=buf.getvalue(),
        file_name='k-means_stereonet_report.png',
        mime='image/png'
    )
    
    # Optional: Display data with new cluster labels
    st.subheader("ข้อมูลพร้อม Joint Set Label")
    st.dataframe(df)

    # 7. ปุ่มดาวน์โหลดข้อมูล CSV
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

# --------------------------------------------------
# Streamlit Main App Interface
# --------------------------------------------------
def main_app():
    st.title("🪨 K-Means Stereonet Plotter (Web App)")
    st.markdown("""
        แอปพลิเคชันนี้ใช้ **K-Means Clustering** เพื่อจัดกลุ่มชุดข้อมูลทิศทาง (Dip Direction & Dip Angle)
        และแสดงผลบน **Stereonet**
    """)
    st.markdown("---")

    # --- INPUT WIDGETS ---
    
    # 1. File Uploader
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
            
            # 2. Sidebar for controls (Optional but cleaner)
            st.sidebar.header("⚙️ การตั้งค่าการ Plot")
            
            # K-Value Slider
            k_value = st.sidebar.slider(
                "2. จำนวนคลัสเตอร์ K (Joint Sets):",
                min_value=2, max_value=10, value=3, step=1,
                help="จำนวนกลุ่มที่ K-Means จะจัด (K=2 ถึง K=10)"
            )
            
            # Plot Type Radio Button
            plot_type = st.sidebar.radio(
                "3. เลือก Plot Type:",
                ("Pole", "Plane"),
                help="Pole: แสดงจุด (เหมาะสำหรับ Clustering), Plane: แสดงเส้นระนาบ"
            )

            # --------------------------------------------------
            # 💡 Advanced Options
            # --------------------------------------------------
            st.sidebar.markdown("---")
            st.sidebar.subheader("ตัวเลือกกราฟขั้นสูง")
            
            # Font Selection
            font_name = st.sidebar.selectbox(
                "4. เลือก Font สำหรับกราฟ:",
                options=["Tahoma", "Arial", "Times New Roman", "DejaVu Sans"],
                index=0, 
                help="Font ที่ใช้ใน Title, Label, และ Legend"
            )
            
            # --- RUN BUTTON ---
            if st.button("🚀 ประมวลผลและแสดง Stereonet"):
                # Run the clustering and plotting with the remaining parameters
                plot_stereonet(df, k_value, plot_type, font_name)
        else:
            st.warning("กรุณาตรวจสอบโครงสร้างไฟล์และลองใหม่อีกครั้ง")


if __name__ == "__main__":
    main_app()