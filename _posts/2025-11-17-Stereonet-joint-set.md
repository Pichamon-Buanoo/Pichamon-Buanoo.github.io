---
layout: post
title: "create joint dataset on stereonet"
date: 2025-11-16
categories: [Education, Geology]
tags: [stereonet,python]
---

### Introduction
Stereographic projection is a widely used tool in various geological applications, including structural geology, failure analysis, and borehole interpretation. When working with a large number of dip-direction and dip-angle measurements from a study area, grouping the data into meaningful joint sets often requires mathematical computation or manual classification by the interpreter.

Therefore, I aimed to explore Python-based methods to automatically classify joint sets by importing data from a .csv file and generating stereonet visualizations with color-coded clusters.
- [Stereonet](/images/stereonet_output.png)<img src='/images/stereonet_output.png'>

### Download
- [Download Python script](https://Pichamon-Buanoo.github.io/files/kmean.py)
- [Download data](https://Pichamon-Buanoo.github.io/files/joint.csv)

### Python code
```python
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
    print(df[df["joint_set"] == i])```
