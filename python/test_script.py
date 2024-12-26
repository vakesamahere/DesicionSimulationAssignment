import math

def combination(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

# 计算 C(17, 10)
s= 0
for i in range(11):
    result = combination(17, i)
    print(i,result)
    s+=result

print('sum',s)