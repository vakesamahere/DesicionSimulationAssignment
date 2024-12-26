from test import *
import pandas as pd
rows = []
with open(f'log.txt','w',encoding='utf-8') as f:
    f.write(f"{time.time():2f}")
# for i in range(0,len(dorms)+1):
for i in range(0,max_load+1):
    simulation_index = i
    rows_temp = inner_main(i)
    # rows.append(rows_temp)
    rows=rows+rows_temp
    
df = pd.DataFrame(rows)
# 将DataFrame写入Excel文件
df.to_excel('output.xlsx', index=False, header=True)

