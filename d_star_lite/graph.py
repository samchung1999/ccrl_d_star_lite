import heapq  # 引入堆队列，用于优先队列
from utils import stateNameToCoords  # 引入工具函数，将状态名转换为坐标

class Node:
    def __init__(self, id):
        self.id = id
        self.parents = {}  # 父节点字典，键为父节点ID，值为（边的代价）
        self.children = {}  # 子节点字典，键为子节点ID，值为（边的代价）
        self.g = float('inf')  # g值，表示从起点到该节点的最短路径估计
        self.rhs = float('inf')  # rhs值，表示从该节点到目标的最短路径估计

    def __str__(self):
        return 'Node: ' + self.id + ' g: ' + str(self.g) + ' rhs: ' + str(self.rhs)

    def __repr__(self):
        return self.__str__()

    def update_parents(self, parents):
        self.parents = parents  # 更新父节点字典

class Graph:
    def __init__(self):
        self.graph = {}  # 图字典，键为节点ID，值为Node对象

    def __str__(self):
        msg = 'Graph:'
        for i in self.graph:
            msg += '\n  node: ' + i + ' g: ' + str(self.graph[i].g) + ' rhs: ' + str(self.graph[i].rhs)
        return msg

    def __repr__(self):
        return self.__str__()

    def setStart(self, id):
        if id in self.graph:
            self.start = id  # 设置起点
        else:
            raise ValueError('start id not in graph')

    def setGoal(self, id):
        if id in self.graph:
            self.goal = id  # 设置终点
        else:
            raise ValueError('goal id not in graph')

def addNodeToGraph(graph, id, neighbors, edge=1):
    node = Node(id)
    for i in neighbors:
        node.parents[i] = edge  # 将邻居节点添加到父节点字典
        node.children[i] = edge  # 将邻居节点添加到子节点字典
    graph[id] = node  # 将节点添加到图中
    return graph

def makeGraph():
    graph = {}

    # 构建4连通图（无对角线）
    graph = addNodeToGraph(graph, 'x1y1', ['x1y2', 'x2y1'])
    graph = addNodeToGraph(graph, 'x2y1', ['x1y1', 'x3y1', 'x2y2'])
    graph = addNodeToGraph(graph, 'x1y2', ['x1y1', 'x2y2'])
    graph = addNodeToGraph(graph, 'x2y2', ['x1y2', 'x2y1', 'x3y2'])
    graph = addNodeToGraph(graph, 'x3y1', ['x3y2', 'x2y1'])
    graph = addNodeToGraph(graph, 'x3y2', ['x3y1', 'x2y2'])

    return graph
