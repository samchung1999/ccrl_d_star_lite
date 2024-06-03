from graph import Node, Graph  # 从graph模块导入Node和Graph类

class GridWorld(Graph):
    def __init__(self, x_dim, y_dim, connect8=True):
        self.x_dim = x_dim  # 网格的宽度
        self.y_dim = y_dim  # 网格的高度
        self.cells = [0] * y_dim  # 初始化网格，每行一个元素
        for i in range(y_dim):
            self.cells[i] = [0] * x_dim  # 将每个元素替换为行，行的长度为网格的宽度
        self.connect8 = connect8  # 是否为8连通图
        self.graph = {}  # 初始化图字典

        self.generateGraphFromGrid()  # 从网格生成图

    def __str__(self):
        msg = 'Graph:'
        for i in self.graph:
            msg += '\n  node: ' + i + ' g: ' + str(self.graph[i].g) + ' rhs: ' + str(self.graph[i].rhs) + \
                   ' neighbors: ' + str(self.graph[i].children)
        return msg

    def __repr__(self):
        return self.__str__()

    def printGrid(self):
        print('** GridWorld **')
        for row in self.cells:
            print(row)

    def printGValues(self):
        for j in range(self.y_dim):
            str_msg = ""
            for i in range(self.x_dim):
                node_id = 'x' + str(i) + 'y' + str(j)
                node = self.graph[node_id]
                if node.g == float('inf'):
                    str_msg += ' - '
                else:
                    str_msg += ' ' + str(node.g) + ' '
            print(str_msg)

    def generateGraphFromGrid(self):
        edge = 1
        for i in range(len(self.cells)):
            row = self.cells[i]
            for j in range(len(row)):
                node = Node('x' + str(i) + 'y' + str(j))  # 创建节点
                if i > 0:  # 非顶部行
                    node.parents['x' + str(i - 1) + 'y' + str(j)] = edge
                    node.children['x' + str(i - 1) + 'y' + str(j)] = edge
                if i + 1 < self.y_dim:  # 非底部行
                    node.parents['x' + str(i + 1) + 'y' + str(j)] = edge
                    node.children['x' + str(i + 1) + 'y' + str(j)] = edge
                if j > 0:  # 非最左列
                    node.parents['x' + str(i) + 'y' + str(j - 1)] = edge
                    node.children['x' + str(i) + 'y' + str(j - 1)] = edge
                if j + 1 < self.x_dim:  # 非最右列
                    node.parents['x' + str(i) + 'y' + str(j + 1)] = edge
                    node.children['x' + str(i) + 'y' + str(j + 1)] = edge
                self.graph['x' + str(i) + 'y' + str(j)] = node  # 将节点添加到图中
