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
### ไฟล์แนบดาวน์โหลด
- [Download Python script](https://Pichamon-Buanoo.github.io/files/kmean.py)
- [Download data](https://Pichamon-Buanoo.github.io/files/joint.csv)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
from sklearn.cluster import KMeans
df = pd.read_csv("joint.csv")

![result](path/to/stereonet_output.png)
