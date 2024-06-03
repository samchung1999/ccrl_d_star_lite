import numpy as np  # 匯入 NumPy 庫，用於數值計算
import pandas as pd  # 匯入 Pandas 庫，用於數據處理
import seaborn as sns  # 匯入 Seaborn 庫，用於數據可視化
import matplotlib.pyplot as plt  # 匯入 Matplotlib 庫，用於繪圖
import os  # 匯入 OS 庫，用於操作系統相關操作
import sys  # 匯入 sys 庫，用於系統相關操作

# 獲取當前腳本的絕對路徑
abspath = os.path.dirname(os.path.abspath(__file__))

# 設置地圖名稱
map_name = 'test_map_l1'

# 設置 Excel 文件路徑
excel_path = abspath + './{}.xlsx'.format(map_name)

# 讀取 Excel 文件，並將其內容轉換為 DataFrame
raw_data = pd.read_excel(r'{}'.format(excel_path), header=None)

# 將 DataFrame 轉換為 NumPy 數組
map = np.array(raw_data.values, dtype=np.uint8)

# 打印地圖的尺寸
print("map_size={}".format(map.shape))

# 將地圖保存為 NumPy 二進制文件
np.save('./{}.npy'.format(map_name), map)

# 設置繪圖的尺寸
plt.figure(figsize=(6, 6))

# 使用 Seaborn 繪製地圖的熱圖
sns.heatmap(map, cmap='Greys', cbar=False)

# 顯示繪圖
plt.show()
