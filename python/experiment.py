from itertools import combinations
import statistics
import pandas as pd
from test import dorms,max_load

for combo in combinations([i for i in range(10)], 6):
    print(combo)