import heapq
from itertools import permutations
import sys
import pandas as pd
import numpy as np
import time

# 原始图的定义
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

# 创建双向图
graph = {}
for node in original_graph:
    if node not in graph:
        graph[node] = {}
    for neighbor, weight in original_graph[node].items():
        graph[node][neighbor] = weight
        if neighbor not in graph:
            graph[neighbor] = {}
        graph[neighbor][node] = weight

def dijkstra(graph, start, end, n=32):
    """
    基础的Dijkstra算法，用于计算两点间的最短路径
    """
    distances = {str(i): float('inf') for i in range(n)}
    distances[str(start)] = 0
    pq = [(0, start, [start])]  # (距离, 当前节点, 路径)
    visited = set()

    while pq:
        current_distance, current, path = heapq.heappop(pq)
        
        if str(current) == str(end):
            return current_distance, path
            
        if str(current) in visited:
            continue
            
        visited.add(str(current))

        if str(current) not in graph:
            continue

        for neighbor, weight in graph[str(current)].items():
            if neighbor in visited:
                continue
            
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                new_path = path + [int(neighbor)]
                heapq.heappush(pq, (distance, int(neighbor), new_path))
    
    return float('inf'), []

def find_optimal_path_with_required_nodes(graph, start, end, required_nodes):
    """
    找到包含必经点的最短路径
    """
    # 确保起点和终点不在必经点列表中
    required_nodes = [node for node in required_nodes if node != start and node != end]
    
    # 计算所有必经点之间（包括起点和终点）的最短路径
    all_points = [start] + required_nodes + [end]
    n = len(all_points)
    
    # 创建距离矩阵和路径矩阵
    distances = {}
    paths = {}
    # print("\n计算节点间距离:")
    for i in range(n):
        for j in range(n):
            if i != j:
                dist, path = dijkstra(graph, all_points[i], all_points[j])
                distances[(all_points[i], all_points[j])] = dist
                paths[(all_points[i], all_points[j])] = path
                # print(f"从节点 {all_points[i]} 到节点 {all_points[j]}: 距离={dist}")
    
    # 检查是否存在无法到达的点
    for i in range(n):
        for j in range(n):
            if i != j and distances[(all_points[i], all_points[j])] == float('inf'):
                print(f"\n警告: 从节点 {all_points[i]} 无法到达节点 {all_points[j]}")
                return float('inf'), []
    
    # 动态规划状态和路径记录
    dp = {}
    path_dp = {}
    
    def solve_dp(current, visited):
        if visited == (1 << len(required_nodes)) - 1:
        # if 1 == len(required_nodes):
            return distances[(current, end)], paths[(current, end)]
        
        state = (current, visited)
        if state in dp:
            return dp[state], path_dp[state]
        
        min_dist = float('inf')
        best_path = []
        
        # 尝试访问每个未访问的必经点
        for i in range(len(required_nodes)):
            if not (visited & (1 << i)): # 当且仅当visited[i]==1时通过
                next_node = required_nodes[i]
                dist_to_next = distances[(current, next_node)]
                remaining_dist, remaining_path = solve_dp(next_node, visited | (1 << i)) # 将节点i设为1
                total_dist = dist_to_next + remaining_dist
                
                if total_dist < min_dist:
                    min_dist = total_dist
                    best_path = paths[(current, next_node)][:-1] + remaining_path
        
        dp[state] = min_dist
        path_dp[state] = best_path
        return min_dist, best_path
    
    # 从起点开始求解
    final_dist, final_path = solve_dp(start, 0)
    
    return final_dist, final_path

def cuntomized_find_optimal_path_with_required_nodes(graph, start, end, required_nodes):
    """
    找到包含必经点的最短路径
    """
    # 确保起点和终点不在必经点列表中
    required_nodes = [node for node in required_nodes if node != start and node != end]
    
    # 计算所有必经点之间（包括起点和终点）的最短路径
    all_points = [start] + required_nodes + [end]
    n = len(all_points)
    
    # 创建距离矩阵和路径矩阵
    distances = {}
    paths = {}
    # print("\n计算节点间距离:")
    for i in range(n):
        for j in range(n):
            if i != j:
                dist, path = dijkstra(graph, all_points[i], all_points[j])
                distances[(all_points[i], all_points[j])] = dist
                paths[(all_points[i], all_points[j])] = path
                # print(f"从节点 {all_points[i]} 到节点 {all_points[j]}: 距离={dist}")
    
    # 检查是否存在无法到达的点
    for i in range(n):
        for j in range(n):
            if i != j and distances[(all_points[i], all_points[j])] == float('inf'):
                print(f"\n警告: 从节点 {all_points[i]} 无法到达节点 {all_points[j]}")
                return float('inf'), []
    
    # 动态规划状态和路径记录
    dp = {}
    path_dp = {}
    
    def solve_dp(current, visited):
        if visited == (1 << len(required_nodes)) - 1:
        # if 1 == len(required_nodes):
            return distances[(current, end)], paths[(current, end)]
        
        state = (current, visited)
        if state in dp:
            return dp[state], path_dp[state]
        
        min_dist = float('inf')
        best_path = []
        
        # 尝试访问每个未访问的必经点
        for i in range(len(required_nodes)):
            if not (visited & (1 << i)): # 当且仅当visited[i]==1时通过
                next_node = required_nodes[i]
                dist_to_next = distances[(current, next_node)]
                remaining_dist, remaining_path = solve_dp(next_node, visited | (1 << i)) # 将节点i设为1
                total_dist = dist_to_next + remaining_dist
                
                if total_dist < min_dist:
                    min_dist = total_dist
                    best_path = paths[(current, next_node)][:-1] + remaining_path
        
        dp[state] = min_dist
        path_dp[state] = best_path
        return min_dist, best_path
    
    # 从起点开始求解
    final_dist, final_path = solve_dp(start, 0)
    
    return final_dist, final_path

def print_path_details(distance, path, graph):
    """
    打印详细的路径信息
    """
    if distance == float('inf'):
        print("没有找到有效路径！")
        return
    
    print(f"\n总距离: {distance:.2f}")
    print("\n详细路径:")
    
    total_dist = 0
    for i in range(len(path)-1):
        current = str(path[i])
        next_node = str(path[i+1])
        segment_dist = graph[current][next_node]
        total_dist += segment_dist
        print(f"{path[i]} -> {path[i+1]}: {segment_dist:.2f}")
    
    print(f"\n路径节点序列: {' -> '.join(map(str, path))}")
    print(f"验证总距离: {total_dist:.2f}")

def get_path_details(distance, path, graph):
    """
    输出详细的路径信息
    """
    if distance == float('inf'):
        return (float('inf'),"没有找到有效路径！")
    
    # print(f"\n总距离: {distance:.2f}")
    # print("\n详细路径:")
    
    total_dist = 0
    for i in range(len(path)-1):
        current = str(path[i])
        next_node = str(path[i+1])
        segment_dist = graph[current][next_node]
        total_dist += segment_dist
    return total_dist,path

def test_connectivity():
    """
    测试图的连通性s
    """
    print("\n测试图的连通性:")
    start = '0'
    end = '31'
    dist, path = dijkstra(graph, int(start), int(end))
    if dist == float('inf'):
        print(f"从节点 {start} 到节点 {end} 不存在路径")
    else:
        print(f"从节点 {start} 到节点 {end} 存在路径，距离为: {dist}")
        print(f"路径为: {' -> '.join(map(str, path))}")

def create_distance_matrix():
    """
    创建并保存距离矩阵到CSV文件
    """
    df = pd.DataFrame(index=range(32), columns=range(32))
    
    # 用字典中的数据填充DataFrame
    for node1 in graph:
        for node2, distance in graph[node1].items():
            i, j = int(node1), int(node2)
            df.iloc[i, j] = distance
    
    # 将NaN值替换为空字符串
    df = df.fillna('')
    
    # 保存为CSV文件
    df.to_csv('distance_matrix.csv')
    print("\n距离矩阵已保存到 'distance_matrix.csv'")



def main():
    # 创建距离矩阵
    create_distance_matrix()
    
    # 测试图的连通性
    test_connectivity()
    
    # 测试不同的场景
    test_cases = [
        # {
        #     'name': "测试场景1 - 单个必经点",
        #     'start': 0,
        #     'end': 0,
        #     'required': [10,12,16]
        # },
        # {
        #     'name': "测试场景2 - 两个必经点",
        #     'start': 0,
        #     'end': 30,
        #     'required': [1, 20,8]
        # },
        {
            'name': "测试场景3 - 三个必经点",
            'start': 0,
            'end': 0,
            # 'required': [i for i in range(1,32)]
            'required': [3,5,6,7,8,9,13,14,15,16,21,24,25,27,28,29,31]
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"起点: {case['start']}")
        print(f"终点: {case['end']}")
        print(f"必经点: {case['required']}")
        start = time.time()
        distance, path = find_optimal_path_with_required_nodes(
            graph, 
            case['start'], 
            case['end'], 
            case['required']
        )
        
        end = time.time()
        print_path_details(distance, path, graph)
        print(f"用时{end-start}")

if __name__ == "__main__":
    main()