import numpy as np
import cv2
import gym
from gym import spaces
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from d_star_lite import initDStarLite, moveAndRescan
from grid import GridWorld
from utils import stateNameToCoords

abspath = os.path.dirname(os.path.abspath(__file__))

class ExploreEnv_v1(gym.Env):
    def __init__(self, map_name='train_map_l1', random_obstacle=False, training=False, render=False):
        self.explore_map_size = 24
        self.local_map_size_half = 12
        self.episode_limit = 300
        self.occupied_pixel = 255
        self.unknown_pixel = 128
        self.free_pixel = 0
        self.explored_pixel = 255
        self.agent_pixel = 128
        self.laser_range_max = 50
        self.laser_angle_resolution = 0.05 * np.pi
        self.laser_angle_half = 0.75 * np.pi
        self.orientation = 0
        self.position = np.zeros(2, dtype=np.int32)
        self.move = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        self.explore_rate = 0
        self.episode_steps = 0

        self.ground_truth_map = np.load(abspath + '/map/{}.npy'.format(map_name))
        self.ground_truth_map.flags.writeable = False
        self.real_map = deepcopy(self.ground_truth_map)
        self.map = np.ones_like(self.real_map) * self.unknown_pixel
        self.grid_num = (self.real_map != self.unknown_pixel).sum()
        self.global_map = np.zeros((self.explore_map_size, self.explore_map_size), dtype=np.uint8)
        self.local_map = np.zeros((self.explore_map_size, self.explore_map_size), dtype=np.uint8)

        self.random_obstacle = random_obstacle
        self.num_obstacles = 4
        if training:
            self.max_explore_rate = 0.99
        else:
            self.max_explore_rate = 1.0

        self.action_dim = 2
        self.s_map_dim = (2, self.explore_map_size, self.explore_map_size)
        self.s_sensor_dim = (round(2 * self.laser_angle_half / self.laser_angle_resolution) + 2,)
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.explore_map_size - 1, self.explore_map_size - 1]), dtype=np.int32)
        self.observation_space = spaces.Dict({
            "s_map": spaces.Box(low=0, high=255, shape=self.s_map_dim, dtype=np.uint8),
            "s_sensor": spaces.Box(low=0, high=1.0, shape=self.s_sensor_dim, dtype=np.float32)
        })

        self.graph = GridWorld(self.explore_map_size, self.explore_map_size, connect8=False)
        self.queue = []
        self.k_m = 0
        self.graph, self.queue, self.k_m = initDStarLite(self.graph, self.queue, 'x0y0', 'x23y23', self.k_m)

        print("init {}".format(map_name))

    def update_map(self):
        self.map[self.position[0], self.position[1]] = self.real_map[self.position[0], self.position[1]]
        laser = []
        for theta in np.arange(self.orientation * 0.5 * np.pi - self.laser_angle_half, self.orientation * 0.5 * np.pi + self.laser_angle_half + 1e-5, self.laser_angle_resolution):
            for r in range(1, self.laser_range_max + 1):
                dim0 = int(round(self.position[0] + r * np.sin(theta)))
                dim1 = int(round(self.position[1] + r * np.cos(theta)))
                self.map[dim0, dim1] = self.real_map[dim0, dim1]
                if self.real_map[dim0, dim1] == self.occupied_pixel:
                    break
            laser.append(np.sqrt((dim0 - self.position[0]) ** 2 + (dim1 - self.position[1]) ** 2))
        return np.array(laser, dtype=np.float32)

    def get_state(self):
        laser = self.update_map()
        self.local_map = self.map[self.position[0] - self.local_map_size_half:self.position[0] + self.local_map_size_half, self.position[1] - self.local_map_size_half:self.position[1] + self.local_map_size_half]
        explore_map = (self.map != self.unknown_pixel) * self.explored_pixel
        explore_rate = explore_map.sum() / (self.grid_num * self.explored_pixel)
        nonzero_index = np.nonzero(explore_map)
        dim0_min = nonzero_index[0].min()
        dim0_max = nonzero_index[0].max()
        dim1_min = nonzero_index[1].min()
        dim1_max = nonzero_index[1].max()
        global_map = explore_map[dim0_min:dim0_max + 1, dim1_min:dim1_max + 1]
        global_map = cv2.resize(global_map, dsize=(self.explore_map_size, self.explore_map_size), interpolation=cv2.INTER_NEAREST)
        position_0 = int((self.position[0] - dim0_min) * self.explore_map_size / (dim0_max - dim0_min))
        position_1 = int((self.position[1] - dim1_min) * self.explore_map_size / (dim1_max - dim1_min))
        global_map[np.clip(position_0 - 1, 0, self.explore_map_size):np.clip(position_0 + 2, 0, self.explore_map_size),
                   np.clip(position_1 - 1, 0, self.explore_map_size):np.clip(position_1 + 2, 0, self.explore_map_size)] = self.agent_pixel
        self.global_map = global_map.astype(np.uint8)
        s_map = np.stack([self.global_map, self.local_map], axis=0)
        s_sensor = np.concatenate([laser / self.laser_range_max, np.array([self.orientation / 4], dtype=np.float32)])
        s = {"s_map": s_map, "s_sensor": s_sensor}
        return s, explore_rate

    def get_info(self):
        return {"explore_rate": self.explore_rate, "position": self.position, 'episode_steps': self.episode_steps}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        if self.random_obstacle:
            self.random_init_obstacle()
        self.random_init_agent()
        self.map = np.ones_like(self.real_map) * self.unknown_pixel
        s, explore_rate = self.get_state()
        self.explore_rate = explore_rate
        info = self.get_info()
        return s, info

    def step(self, action):
        self.episode_steps += 1
        self.goal_position = stateNameToCoords(action)
        s_new, self.k_m = moveAndRescan(self.graph, self.queue, 'x{}y{}'.format(self.position[0], self.position[1]), 3, self.k_m)
        if s_new == 'goal':
            s_new = action
        self.position = np.array(stateNameToCoords(s_new))
        s, explore_rate = self.get_state()
        if explore_rate > self.explore_rate:
            r = np.clip((explore_rate ** 2 - self.explore_rate ** 2) * 10, 0, 1.0)
        else:
            r = -0.005
        terminal = explore_rate >= self.max_explore_rate or self.episode_steps == self.episode_limit
        self.explore_rate = explore_rate
        info = self.get_info()
        return s, r, terminal, False, info

    def render(self, mode='human'):
        plt.subplot(2, 2, 1)
        sns.heatmap(self.real_map, cmap='Greys', cbar=False)  # 绘制真实地图
        plt.scatter(self.position[1] + 0.5, self.position[0] + 0.5, c='r', marker='s', s=50)  # 标记当前位置
        plt.subplot(2, 2, 2)
        sns.heatmap(self.map, cmap='Greys', cbar=False)  # 绘制当前已知地图
        plt.scatter(self.position[1] + 0.5, self.position[0] + 0.5, c='r', marker='s', s=50)  # 标记当前位置
        plt.subplot(2, 2, 3)
        sns.heatmap(self.local_map, cmap='Greys', cbar=False)  # 绘制局部地图
        plt.subplot(2, 2, 4)
        sns.heatmap(self.global_map, cmap='Greys', cbar=False)  # 绘制全局地图
        plt.show()  # 显示绘图
        plt.pause(0.05)  # 暂停以更新图像
