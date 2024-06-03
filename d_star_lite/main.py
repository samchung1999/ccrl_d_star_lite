import heapq
import pygame
from graph import Node, Graph
from grid import GridWorld
from utils import stateNameToCoords
from d_star_lite import initDStarLite, moveAndRescan


# 定义一些颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY1 = (145, 145, 102)
GRAY2 = (77, 77, 51)
BLUE = (0, 0, 80)

colors = {
    0: WHITE,
    1: GREEN,
    -1: GRAY1,
    -2: GRAY2
}

# 设置每个网格单元的宽度和高度
WIDTH = 40
HEIGHT = 40

# 设置每个单元之间的边距
MARGIN = 5

# 创建一个二维数组
grid = []
for row in range(10):
    grid.append([])
    for column in range(10):
        grid[row].append(0)  # 添加一个单元

# 设置特定单元的值
grid[1][5] = 1

# 初始化Pygame
pygame.init()

X_DIM = 12
Y_DIM = 12
VIEWING_RANGE = 3

# 设置屏幕的高度和宽度
WINDOW_SIZE = [(WIDTH + MARGIN) * X_DIM + MARGIN,
               (HEIGHT + MARGIN) * Y_DIM + MARGIN]
screen = pygame.display.set_mode(WINDOW_SIZE)

# 设置屏幕标题
pygame.display.set_caption("D* Lite Path Planning")

# 循环直到用户点击关闭按钮
done = False

# 用于管理屏幕更新速度
clock = pygame.time.Clock()

if __name__ == "__main__":
    graph = GridWorld(X_DIM, Y_DIM)
    s_start = 'x1y2'
    s_goal = 'x5y4'
    goal_coords = stateNameToCoords(s_goal)

    graph.setStart(s_start)
    graph.setGoal(s_goal)
    k_m = 0
    s_last = s_start
    queue = []

    graph, queue, k_m = initDStarLite(graph, queue, s_start, s_goal, k_m)

    s_current = s_start
    pos_coords = stateNameToCoords(s_current)

    basicfont = pygame.font.SysFont('Comic Sans MS', 36)

    # -------- 主程序循环 -----------
    while not done:
        for event in pygame.event.get():  # 用户进行某些操作
            if event.type == pygame.QUIT:  # 如果用户点击关闭
                done = True  # 标记我们完成了，退出循环
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                s_new, k_m = moveAndRescan(
                    graph, queue, s_current, VIEWING_RANGE, k_m)
                if s_new == 'goal':
                    print('Goal Reached!')
                    done = True
                else:
                    s_current = s_new
                    pos_coords = stateNameToCoords(s_current)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 用户点击鼠标，获取位置
                pos = pygame.mouse.get_pos()
                # 将屏幕坐标转换为网格坐标
                column = pos[0] // (WIDTH + MARGIN)
                row = pos[1] // (HEIGHT + MARGIN)
                # 设置该位置为障碍物
                if(graph.cells[row][column] == 0):
                    graph.cells[row][column] = -1

        # 设置屏幕背景
        screen.fill(BLACK)

        # 绘制网格
        for row in range(Y_DIM):
            for column in range(X_DIM):
                color = WHITE
                pygame.draw.rect(screen, colors[graph.cells[row][column]],
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])
                node_name = 'x' + str(column) + 'y' + str(row)
                if(graph.graph[node_name].g != float('inf')):
                    text = basicfont.render(
                        str(graph.graph[node_name].g), True, (0, 0, 200))
                    textrect = text.get_rect()
                    textrect.centerx = int(
                        column * (WIDTH + MARGIN) + WIDTH / 2) + MARGIN
                    textrect.centery = int(
                        row * (HEIGHT + MARGIN) + HEIGHT / 2) + MARGIN
                    screen.blit(text, textrect)

        # 将目标单元填充为绿色
        pygame.draw.rect(screen, GREEN, [(MARGIN + WIDTH) * goal_coords[0] + MARGIN,
                                         (MARGIN + HEIGHT) * goal_coords[1] + MARGIN, WIDTH, HEIGHT])
        # 绘制机器人的当前位置
        robot_center = [int(pos_coords[0] * (WIDTH + MARGIN) + WIDTH / 2) +
                        MARGIN, int(pos_coords[1] * (HEIGHT + MARGIN) + HEIGHT / 2) + MARGIN]
        pygame.draw.circle(screen, RED, robot_center, int(WIDTH / 2) - 2)

        # 绘制机器人的视野范围
        pygame.draw.rect(
            screen, BLUE, [robot_center[0] - VIEWING_RANGE * (WIDTH + MARGIN), robot_center[1] - VIEWING_RANGE * (HEIGHT + MARGIN), 2 * VIEWING_RANGE * (WIDTH + MARGIN), 2 * VIEWING_RANGE * (HEIGHT + MARGIN)], 2)

        # 限制每秒帧数
        clock.tick(20)

        # 更新屏幕显示
        pygame.display.flip()

    pygame.quit()
