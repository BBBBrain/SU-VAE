"""2023 / 8 / 12
18: 01
dell"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# 读取Excel文件
from matplotlib import colors
import math
dataframe = pd.read_excel(r'C:\Users\dell\Desktop\\gene_all.xlsx',sheet_name="BA23")

# 获取数据
names = dataframe['name'].tolist()[:20]
names2 = []
for i in range(len(names)):
    names2.append(names[i])
    names2.append("q")
one_dim = dataframe['FDR'].tolist()[:20]
one_dim2 = []
for i in range(len(one_dim)):
    one_dim2.append(one_dim[i])
    one_dim2.append(0)
one_dim = [- math.log10(i) for i in one_dim]
color_dim =  dataframe['Fold Enrichment'].tolist()[:20]
color_dim2= []
for i in range(len(color_dim)):
    color_dim2.append(color_dim[i])
    color_dim2.append(0)

colormap = cm.ScalarMappable(cmap='coolwarm',norm=colors.Normalize(vmin=min(one_dim), vmax=max(one_dim)))
colormap.set_array([])  # Set an empty array for the colormap
print(names2)
# 绘制图形
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(names, color_dim, color=[colormap.to_rgba(val) for val in one_dim])

plt.xticks(list(range(0,len(names))), names, rotation=30, ha='right')
print(len(names))
print(np.linspace(0,len(names),1))
# ax.set_xlim(np.linspace(0,len(names),1))


# Adjust tick params to increase spacing between ticks

plt.subplots_adjust(bottom=0.5)  # Adjust as needed
plt.subplots_adjust(left=0.35)  # Adjust as needed
cbar = plt.colorbar(colormap, ax=ax)
cbar.set_label('-log10(FDR)')

# plt.xticks(rotation=45, ha='left')
# 设置刻度和标签
ax.set_ylabel('Fold Enrichment')
# ax.set_xlabel('Names')
plt.title("gene enrichment on BA23")
# 显示图形
plt.show()