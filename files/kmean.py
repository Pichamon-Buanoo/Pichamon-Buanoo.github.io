import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
from sklearn.cluster import KMeans

df = pd.read_csv("joint.csv")

df.columns = df.columns.str.lower().str.strip()

dip_dir = df["dip direction"].values 
dip = df["dip angle"].values

def orientation_to_vector(dd, dip):
    dd_rad = np.radians(dd)
    dip_rad = np.radians(dip)
    
    strike = (dd - 90) % 360
    strike_rad = np.radians(strike)

    nx = np.sin(dip_rad) * np.sin(strike_rad)
    ny = np.sin(dip_rad) * np.cos(strike_rad)
    nz = np.cos(dip_rad)
    return [nx, ny, nz]

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
    ax.plane(subset["dip direction"], subset["dip angle"], color=colors[i], label=f"Set {i}")

ax.legend()
plt.savefig('stereonet_output.png', dpi=300)

for i in range(k):
    print(f"\n-------- Joint Set {i} --------")
    print(df[df["joint_set"] == i])
