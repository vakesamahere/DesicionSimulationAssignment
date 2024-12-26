original_graph = {
    '0': {'1': 0.57},
    '1': {'0': 0.57, '2': 1.14, '18': 0.57},
    '2': {'1': 1.14, '3': 3.84,'29': 14.1},
    '3': {'2': 3.84, '4': 5.89},
    '4': {'3': 5.89, '5': 1.76, '12': 3.76, '16': 3.39},
    '5': {'4': 1.76, '6': 1.53},
    '6': {'5': 1.53, '7': 1.76},
    '7': {'6': 1.76, '8': 0.57},
    '8': {'7': 0.57, '9': 0.73},
    '9': {'8': 0.73, '10': 2.58},
    '10': {'9': 2.58, '11': 3.25, '19': 1.79},
    '11': {'10': 3.25, '12': 9.29},
    '12': {'11': 9.29, '13': 1.23, '4': 3.76,'16': 3.39},
    '13': {'12': 1.23, '14': 2.93},
    '14': {'13': 2.93, '15': 1.71},
    '15': {'14': 1.71, '16': 1.56},
    '16': {'12': 3.39, '15': 1.56},
    '17': {'18': 1.17, '10': 3.96},
    '18': {'17': 1.17, '1': 0.57},
    '19': {'23': 2.86, '20': 4.31, '10': 1.79},
    '20': {'21': 1.87, '2': 4.05},
    '21': {'20': 1.87, '22': 0.98},
    '22': {'21': 0.98, '23': 3.57,'24': 1.61},
    '23': {'22': 3.57, '19': 2.86,'30': 4.23},
    '24': {'23': 1.61, '25': 1.19},
    '25': {'24': 1.19, '26': 1.09},
    '26': {'25': 1.09, '27': 0.79},
    '27': {'26': 0.79, '28': 1.1},
    '28': {'27': 1.1, '29': 1.14},
    '29': {'28': 1.14, '2': 14.1},
    '30': {'26': 3.57, '31': 1.37},
    '31': {'30': 1.37}
}

import numpy as np

# 定义图的顶点数
n = 32

# 初始化权重矩阵
weights = np.full((n, n), np.inf)
for i in range(n):
    weights[i, i] = 0
for i, neighbors in original_graph.items():
    for j, weight in neighbors.items():
        weights[int(i), int(j)] = weight

# 初始化前驱矩阵
parents = np.full((n, n), -1, dtype=int)
for i, neighbors in original_graph.items():
    for j in neighbors:
        parents[int(i), int(j)] = int(i)

# 应用Floyd算法
for k in range(n):
    for i in range(n):
        for j in range(n):
            if weights[i, j] > weights[i, k] + weights[k, j]:
                weights[i, j] = weights[i, k] + weights[k, j]
                parents[i, j] = parents[k, j]

# 输出权重矩阵和前驱矩阵
print("权重矩阵：")
print(weights)
print("前驱矩阵：")
print(parents)

import pandas as pd

columns = [f"{i:02d}" for i in range(n)]
index = [f"{i:02d}" for i in range(n)]

# 填充weights和parents矩阵的代码在这里省略，假设已经填充完毕

# 将numpy数组转换为pandas DataFrame
weights_df = pd.DataFrame(weights, index=index, columns=columns)
parents_df = pd.DataFrame(parents, index=index, columns=columns)

# 保存到Excel文件
with pd.ExcelWriter('shortest_paths.xlsx') as writer:
    weights_df.to_excel(writer, sheet_name='Weights')
    parents_df.to_excel(writer, sheet_name='Parents')

print("权重矩阵和前驱矩阵已保存到Excel文件。")