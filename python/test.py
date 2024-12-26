index = 10
from 最短路 import *
import math
import statistics
from itertools import combinations

def combination(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

dorms = [2,3,5,6,7,8,9,13,14,15,16,17,21,24,25,27,28,29,31]
max_load = 10

def inner_main(simulation_index=index):
    rows = []
    start = time.time()
    # 创建距离矩阵
    create_distance_matrix()

    # 测试图的连通性
    test_connectivity()

    # 生成所有从0到16中选择10个元素的组合
    total_t = 0
    n=0
    N = combination(len(dorms), simulation_index)
    print(N)
    l=len(str(N))+1
    ts = []
    for combo in combinations(dorms, simulation_index):
        required = list(combo)
        case = {
            'name': "测试场景3 - 三个必经点",
            'start': 0,
            'end': 0,
            'required': required
        }
        start = time.time()
        distance, path = find_optimal_path_with_required_nodes(
            graph, 
            case['start'], 
            case['end'], 
            case['required']
        )

        res = get_path_details(distance=distance,path=path,graph=graph)
        rows.append(res)

        t = time.time() -start
        total_t+=t
        n+=1
        if total_t>0:
            print(f"index={simulation_index}    用时{t:0<4f}s   {n:<{l}}/{N}   剩余时间{(N-n)/(n/total_t):.1f}s")
        ts.append(t)
    print(f"总用时{total_t}s")
    print(f"平均用时{total_t/n}s")

    if True:
        # 如果数据集很大，或者你需要更精确的控制，可以使用 NumPy 库
        data = ts
        res=[]
        # 计算均值
        mean_value_np = np.mean(data)
        res.append(f"均值（Mean）: {mean_value_np}")

        # 计算中位数
        median_value_np = np.median(data)
        res.append(f"中位数（Median）: {median_value_np}")

        # 计算方差
        variance_value_np = np.var(data, ddof=0)  # ddof=0 表示使用总体方差
        res.append(f"方差（Variance）: {variance_value_np}")

        # 计算最大值和最小值
        max_value = max(data)
        res.append(f"最大值（Max）: {max_value}")

        import matplotlib.pyplot as plt

        # 绘制直方图
        plt.hist(data, bins=10)  # bins参数可以指定直方图的柱数或柱的边界值
        plt.title('Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)  # 添加网格线
        plt.savefig(f'img/histogram_{simulation_index}.png')  # 保存图形，
        # plt.show()
        plt.close()

        with open(f'log.txt','a',encoding='utf-8') as f:
            f.write(f'\n\nindex={simulation_index}\n')
            f.write('\n'.join(res))
    return rows