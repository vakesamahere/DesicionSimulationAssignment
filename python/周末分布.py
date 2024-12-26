import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 设置matplotlib的中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 数据
arrivals = [2, 7, 22, 37, 38, 40, 41, 43, 46, 48, 54, 57]  # 到达时间（分钟）
departures = [5, 12, 26, 40, 42, 44, 47, 48, 50, 53, 57, 59]  # 离开时间（分钟）

# 创建数据框
df = pd.DataFrame({
    'arrival': arrivals,
    'departure': departures
})

# 基本统计分析
def analyze_arrivals():
    # 计算到达间隔
    interarrival_times = np.diff(arrivals)
    
    # 分析两个时段（0-30分钟和30-60分钟）
    period1_arrivals = [t for t in arrivals if t < 30]
    period2_arrivals = [t for t in arrivals if t >= 30]
    
    # 计算各时段的到达率（每分钟）
    rate1 = len(period1_arrivals) / 30
    rate2 = len(period2_arrivals) / 30
    
    return {
        'interarrival_times': interarrival_times,
        'rate_period1': rate1,
        'rate_period2': rate2,
        'total_customers': len(arrivals)
    }

# 执行到达分析
analysis = analyze_arrivals()

print(f"基础统计信息：")
print(f"总顾客数：{analysis['total_customers']}")
print(f"17:00-17:30 到达率：{analysis['rate_period1']:.3f} 客户/分钟")
print(f"17:30-18:00 到达率：{analysis['rate_period2']:.3f} 客户/分钟")

# 双队列系统分析
def analyze_queueing_system(arrivals, departures):
    n = len(arrivals)
    queue1 = []  # 存储每个队列中顾客的结束服务时间
    queue2 = []
    
    waiting_times = []
    service_times = []
    queue_assignments = []
    
    # 用于跟踪最大队列长度
    max_queue1_length = 0
    max_queue2_length = 0
    
    for i in range(n):
        # 清理已离开的顾客
        queue1 = [t for t in queue1 if t > arrivals[i]]
        queue2 = [t for t in queue2 if t > arrivals[i]]
        
        # 更新最大队列长度
        max_queue1_length = max(max_queue1_length, len(queue1))
        max_queue2_length = max(max_queue2_length, len(queue2))
        
        # 选择较短的队列
        if len(queue1) <= len(queue2):
            selected_queue = queue1
            queue_assignments.append(1)
        else:
            selected_queue = queue2
            queue_assignments.append(2)
        
        # 计算等待时间
        if len(selected_queue) == 0:
            waiting_time = 0
            service_start = arrivals[i]
        else:
            service_start = max(arrivals[i], selected_queue[-1])
            waiting_time = max(0, service_start - arrivals[i])
        
        # 计算服务时间
        service_time = departures[i] - service_start
        
        # 更新队列
        selected_queue.append(departures[i])
        
        waiting_times.append(waiting_time)
        service_times.append(service_time)
    
    return {
        'waiting_times': np.array(waiting_times),
        'service_times': np.array(service_times),
        'queue_assignments': np.array(queue_assignments),
        'avg_waiting_time': np.mean(waiting_times),
        'avg_service_time': np.mean(service_times),
        'max_waiting_time': np.max(waiting_times),
        'max_queue_length': max(max_queue1_length, max_queue2_length)
    }

# 非齐次泊松过程拟合
def fit_nhpp():
    # 将时间标准化到[0,1]区间
    normalized_times = np.array(arrivals) / 60
    
    # 定义分段强度函数
    def intensity(t, params):
        return params[0] if t < 0.5 else params[1]
    
    # 对数似然函数
    def log_likelihood(params):
        log_lik = 0
        for t in normalized_times:
            log_lik += np.log(intensity(t, params))
        integral = params[0] * 0.5 + params[1] * 0.5
        return log_lik - integral
    
    # 使用最大似然估计
    from scipy.optimize import minimize
    initial_guess = [analysis['rate_period1'], analysis['rate_period2']]
    result = minimize(lambda x: -log_likelihood(x), initial_guess, method='Nelder-Mead')
    
    return result.x

# 检验拟合优度
def test_fit_goodness():
    fitted_rates = fit_nhpp()
    
    # 将时间区间分成几个子区间来进行卡方检验
    # 这里我们分成6个10分钟的区间
    intervals = np.arange(0, 61, 10)
    observed = np.zeros(len(intervals)-1)
    expected = np.zeros(len(intervals)-1)
    
    # 计算观察值（实际每个区间的到达数）
    for i in range(len(intervals)-1):
        observed[i] = sum(1 for t in arrivals if intervals[i] <= t < intervals[i+1])
    
    # 计算期望值（根据拟合的分段泊松过程）
    for i in range(len(intervals)-1):
        if intervals[i+1] <= 30:
            # 第一个时段
            expected[i] = fitted_rates[0] * (intervals[i+1] - intervals[i]) / 60
        elif intervals[i] >= 30:
            # 第二个时段
            expected[i] = fitted_rates[1] * (intervals[i+1] - intervals[i]) / 60
        else:
            # 跨越两个时段的区间
            expected[i] = (fitted_rates[0] * (30 - intervals[i]) + 
                         fitted_rates[1] * (intervals[i+1] - 30)) / 60
    
    # 将期望值转换为实际顾客数，并确保总和相等
    total_customers = len(arrivals)
    expected = expected * total_customers / np.sum(expected)  # 归一化确保总和等于总顾客数
    
    # 进行卡方检验
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'observed': observed,
        'expected': expected,
        'intervals': intervals[:-1]
    }

# 计算队列长度随时间变化
def calculate_queue_lengths(arrivals, departures, queue_assignments):
    events = []
    for i in range(len(arrivals)):
        events.append((arrivals[i], 1, queue_assignments[i]))  # 1表示到达
        events.append((departures[i], -1, queue_assignments[i]))  # -1表示离开
    
    events.sort()
    
    queue1_length = []
    queue2_length = []
    times = []
    q1 = 0
    q2 = 0
    
    for time, event_type, queue_num in events:
        if queue_num == 1:
            q1 += event_type
        else:
            q2 += event_type
            
        times.append(time)
        queue1_length.append(q1)
        queue2_length.append(q2)
    
    return times, queue1_length, queue2_length

# 执行分析并输出结果
# 1. 队列系统分析
queue_analysis = analyze_queueing_system(arrivals, departures)

print("\n双队列系统分析结果：")
print(f"平均等待时间：{queue_analysis['avg_waiting_time']:.2f} 分钟")
print(f"平均服务时间：{queue_analysis['avg_service_time']:.2f} 分钟")
print(f"最大等待时间：{queue_analysis['max_waiting_time']:.2f} 分钟")
print(f"最大队列长度：{queue_analysis['max_queue_length']}")

# 2. 非齐次泊松过程拟合结果
fitted_rates = fit_nhpp()
print("\n非齐次泊松过程拟合结果：")
print(f"17:00-17:30 拟合到达率：{fitted_rates[0]:.3f} 客户/分钟")
print(f"17:30-18:00 拟合到达率：{fitted_rates[1]:.3f} 客户/分钟")

# 3. 拟合优度检验
test_results = test_fit_goodness()
print("\n非齐次泊松过程拟合检验（卡方检验）：")
print(f"卡方统计量：{test_results['chi2_statistic']:.3f}")
print(f"p值：{test_results['p_value']:.3f}")

# 可视化
# 1. 到达分布可视化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(analysis['interarrival_times'], bins='auto', alpha=0.7)
plt.title('到达间隔时间分布')
plt.xlabel('间隔时间（分钟）')
plt.ylabel('频次')

plt.subplot(1, 2, 2)
plt.plot(arrivals, range(1, len(arrivals)+1), 'b-', label='累积到达')
plt.axvline(x=30, color='r', linestyle='--', label='时段分界线')
plt.title('累积到达曲线')
plt.xlabel('时间（分钟）')
plt.ylabel('累积顾客数')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 队列分析可视化
plt.figure(figsize=(15, 5))

# 等待时间分布
plt.subplot(1, 3, 1)
plt.hist(queue_analysis['waiting_times'], bins='auto', alpha=0.7)
plt.title('等待时间分布')
plt.xlabel('等待时间（分钟）')
plt.ylabel('频次')

# 服务时间分布
plt.subplot(1, 3, 2)
plt.hist(queue_analysis['service_times'], bins='auto', alpha=0.7)
plt.title('实际服务时间分布')
plt.xlabel('服务时间（分钟）')
plt.ylabel('频次')

# 队列分配情况
plt.subplot(1, 3, 3)
queue_counts = np.bincount(queue_analysis['queue_assignments'])[1:]
plt.bar(['队列1', '队列2'], queue_counts)
plt.title('队列分配情况')
plt.ylabel('顾客数量')

plt.tight_layout()
plt.show()

# 3. 队列长度随时间变化
times, q1_length, q2_length = calculate_queue_lengths(
    arrivals, departures, queue_analysis['queue_assignments']
)

plt.figure(figsize=(12, 6))
plt.step(times, q1_length, label='队列1', where='post')
plt.step(times, q2_length, label='队列2', where='post')
plt.title('队列长度随时间变化')
plt.xlabel('时间（分钟）')
plt.ylabel('队列长度')
plt.grid(True)
plt.legend()
plt.show()

# 4. 观察值vs期望值的可视化
plt.figure(figsize=(10, 6))
plt.bar(test_results['intervals'], test_results['observed'], 
        alpha=0.5, label='观察值', width=8)
plt.plot(test_results['intervals'], test_results['expected'], 
         'r-', label='期望值', linewidth=2)
plt.title('到达人数：观察值 vs 期望值')
plt.xlabel('时间（分钟）')
plt.ylabel('到达人数')
plt.legend()
plt.grid(True)
plt.show()