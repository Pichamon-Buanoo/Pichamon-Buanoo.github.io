import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
from sklearn.cluster import KMeans

df = pd.read_csv("joint.csv")

# เพื่อป้องกันปัญหา KeyError และรองรับชื่อคอลัมน์ "Dip direction" และ "Dip angle"
df.columns = df.columns.str.lower().str.strip()

# 🛠️ การแก้ไขที่ 2 และ 3: ใช้ชื่อคอลัมน์ที่ถูกต้องและดึงเป็น Array 1 มิติ (ลบวงเล็บเหลี่ยมชั้นนอกออก)
dip_dir = df["dip direction"].values 
dip = df["dip angle"].values

def orientation_to_vector(dd, dip):
    # convert dip direction and dip to 3D vector
    # ตอนนี้ dd และ dip คือค่าตัวเลข (scalar) เดี่ยวๆ
    dd_rad = np.radians(dd)
    dip_rad = np.radians(dip)
    
    strike = (dd - 90) % 360
    strike_rad = np.radians(strike)

    nx = np.sin(dip_rad) * np.sin(strike_rad)
    ny = np.sin(dip_rad) * np.cos(strike_rad)
    nz = np.cos(dip_rad)
    return [nx, ny, nz]

# ตอนนี้ zip(dip_dir, dip) จะส่งค่าตัวเลขเดี่ยว ๆ เข้าไปในฟังก์ชันอย่างถูกต้อง
vectors = np.array([orientation_to_vector(dd, d) for dd, d in zip(dip_dir, dip)])

k = 3  
model = KMeans(n_clusters=k).fit(vectors)
labels = model.labels_
df["joint_set"] = labels

print("\nJoint sets assigned!\n")
print(df)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='stereonet')

colors = ["red", "blue", "green", "purple", "orange"]

for i in range(k):
    subset = df[df["joint_set"] == i]
    # 🛠️ การแก้ไขที่ 4: แก้ไขชื่อคอลัมน์ในส่วน plotting ด้วย
    ax.plane(subset["dip direction"], subset["dip angle"], color=colors[i], label=f"Set {i}")

ax.legend()
plt.savefig('stereonet_output.png', dpi=300) # บันทึกเป็นไฟล์ PNG ความละเอียด 300 dpi
# plt.show() # ลบคอมเมนต์ หรือลบทิ้ง

for i in range(k):
    print(f"\n-------- Joint Set {i} --------")
    print(df[df["joint_set"] == i])