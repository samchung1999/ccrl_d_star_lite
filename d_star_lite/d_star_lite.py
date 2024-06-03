import heapq  # 引入堆队列，用于优先队列
from utils import stateNameToCoords  # 引入工具函数，将状态名转换为坐标

# 返回优先队列中具有最小键值的元素的键值
def topKey(queue):
    queue.sort()  # 将队列按键值排序
    if len(queue) > 0:
        return queue[0][:2]  # 返回最小键值的前两个元素
    else:
        return (float('inf'), float('inf'))  # 队列为空时返回无穷大键值

# 计算启发式值，使用曼哈顿距离
def heuristic_from_s(graph, id, s):
    x_distance = abs(int(id.split('x')[1][0]) - int(s.split('x')[1][0]))
    y_distance = abs(int(id.split('y')[1][0]) - int(s.split('y')[1][0]))
    return max(x_distance, y_distance)

# 计算节点的键值
def calculateKey(graph, id, s_current, k_m):
    return (min(graph.graph[id].g, graph.graph[id].rhs) + heuristic_from_s(graph, id, s_current) + k_m, 
            min(graph.graph[id].g, graph.graph[id].rhs))

# 更新节点的 rhs 值和优先队列中的键值
def updateVertex(graph, queue, id, s_current, k_m):
    s_goal = graph.goal
    if id != s_goal:
        min_rhs = float('inf')
        for i in graph.graph[id].children:
            min_rhs = min(min_rhs, graph.graph[i].g + graph.graph[id].children[i])
        graph.graph[id].rhs = min_rhs
    id_in_queue = [item for item in queue if id in item]
    if id_in_queue != []:
        if len(id_in_queue) != 1:
            raise ValueError('more than one ' + id + ' in the queue!')
        queue.remove(id_in_queue[0])
    if graph.graph[id].rhs != graph.graph[id].g:
        heapq.heappush(queue, calculateKey(graph, id, s_current, k_m) + (id,))

# 计算从起点到目标的最短路径
def computeShortestPath(graph, queue, s_start, k_m):
    while (graph.graph[s_start].rhs != graph.graph[s_start].g) or (topKey(queue) < calculateKey(graph, s_start, s_start, k_m)):
        k_old = topKey(queue)
        u = heapq.heappop(queue)[2]
        if k_old < calculateKey(graph, u, s_start, k_m):
            heapq.heappush(queue, calculateKey(graph, u, s_start, k_m) + (u,))
        elif graph.graph[u].g > graph.graph[u].rhs:
            graph.graph[u].g = graph.graph[u].rhs
            for i in graph.graph[u].parents:
                updateVertex(graph, queue, i, s_start, k_m)
        else:
            graph.graph[u].g = float('inf')
            updateVertex(graph, queue, u, s_start, k_m)
            for i in graph.graph[u].parents:
                updateVertex(graph, queue, i, s_start, k_m)

# 在当前路径上获取下一个节点
def nextInShortestPath(graph, s_current):
    min_rhs = float('inf')
    s_next = None
    if graph.graph[s_current].rhs == float('inf'):
        print('You are stuck')
    else:
        for i in graph.graph[s_current].children:
            child_cost = graph.graph[i].g + graph.graph[s_current].children[i]
            if child_cost < min_rhs:
                min_rhs = child_cost
                s_next = i
        if s_next:
            return s_next
        else:
            raise ValueError('could not find child for transition!')

# 扫描并更新障碍物
def scanForObstacles(graph, queue, s_current, scan_range, k_m):
    states_to_update = {}
    range_checked = 0
    if scan_range >= 1:
        for neighbor in graph.graph[s_current].children:
            neighbor_coords = stateNameToCoords(neighbor)
            states_to_update[neighbor] = graph.cells[neighbor_coords[1]][neighbor_coords[0]]
        range_checked = 1

    while range_checked < scan_range:
        new_set = {}
        for state in states_to_update:
            new_set[state] = states_to_update[state]
            for neighbor in graph.graph[state].children:
                if neighbor not in new_set:
                    neighbor_coords = stateNameToCoords(neighbor)
                    new_set[neighbor] = graph.cells[neighbor_coords[1]][neighbor_coords[0]]
        range_checked += 1
        states_to_update = new_set

    new_obstacle = False
    for state in states_to_update:
        if states_to_update[state] < 0:  # 找到障碍物
            for neighbor in graph.graph[state].children:
                if(graph.graph[state].children[neighbor] != float('inf')):
                    neighbor_coords = stateNameToCoords(state)
                    graph.cells[neighbor_coords[1]][neighbor_coords[0]] = -2
                    graph.graph[neighbor].children[state] = float('inf')
                    graph.graph[state].children[neighbor] = float('inf')
                    updateVertex(graph, queue, state, s_current, k_m)
                    new_obstacle = True
    return new_obstacle

# 移动到下一个节点并重新扫描路径
def moveAndRescan(graph, queue, s_current, scan_range, k_m):
    if(s_current == graph.goal):
        return 'goal', k_m
    else:
        s_last = s_current
        s_new = nextInShortestPath(graph, s_current)
        new_coords = stateNameToCoords(s_new)

        if(graph.cells[new_coords[1]][new_coords[0]] == -1):  # 碰到新的障碍物
            s_new = s_current  # 需要重新扫描和重新规划

        results = scanForObstacles(graph, queue, s_new, scan_range, k_m)
        k_m += heuristic_from_s(graph, s_last, s_new)
        computeShortestPath(graph, queue, s_current, k_m)

        return s_new, k_m

# 初始化 D* Lite 算法
def initDStarLite(graph, queue, s_start, s_goal, k_m):
    graph.graph[s_goal].rhs = 0
    heapq.heappush(queue, calculateKey(graph, s_goal, s_start, k_m) + (s_goal,))
    computeShortestPath(graph, queue, s_start, k_m)

    return (graph, queue, k_m)
