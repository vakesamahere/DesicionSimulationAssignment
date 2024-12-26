from itertools import combinations
import statistics
import pandas as pd
from test import dorms,max_load

rows = []
# dorms = [2,3,5,6,7,8,9,13,14,15,16,17,21,24,25,27,28,29,31]
# max_load = 10
indexes = [2**j for j in range(len(dorms))]
for i in range(0,max_load+1):
    simulation_index = i
    rows_temp = []
    for combo in combinations(indexes, i):
        index = sum(combo)
        rows_temp.append(index)
    rows=rows+rows_temp

df = pd.DataFrame(rows)
df.to_excel('output_index.xlsx', index=False, header=True)
