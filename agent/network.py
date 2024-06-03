import torch
import torch.nn as nn
import math

# 定義正交初始化函數
def orthogonal_init(layer, gain=math.sqrt(2)):
    for name, param in layer.named_parameters():  # 遍歷層中的所有參數
        if 'bias' in name:  # 如果是偏置參數
            nn.init.constant_(param, 0)  # 將偏置初始化為0
        elif 'weight' in name:  # 如果是權重參數
            nn.init.orthogonal_(param, gain=gain)  # 使用正交初始化方法
    return layer  # 返回初始化后的層

# 定義 Actor-Critic 網絡模型
class Actor_Critic(nn.Module):
    def __init__(self, config):
        super(Actor_Critic, self).__init__()  # 調用父類的初始化方法
        
        # 定義地圖層
        self.map_layer = nn.Sequential(
            orthogonal_init(nn.Conv2d(config.s_map_dim[0], 8, kernel_size=5, stride=1, padding=2)),  # 卷積層，從 s_map_dim[0] 到 8 通道
            nn.ReLU(),  # ReLU 激活函數
            nn.MaxPool2d(kernel_size=2),  # 最大池化層

            orthogonal_init(nn.Conv2d(8, 16, kernel_size=3, stride=1)),  # 卷積層，從 8 通道到 16 通道
            nn.ReLU(),  # ReLU 激活函數
            nn.MaxPool2d(kernel_size=2),  # 最大池化層

            nn.Flatten(),  # 展平層
            orthogonal_init(nn.Linear(16 * 5 * 5, config.hidden_dim)),  # 全連接層
            nn.ReLU(),  # ReLU 激活函數
        )

        # 定義感測層
        self.sensor_layer = nn.Sequential(
            orthogonal_init(nn.Linear(config.s_sensor_dim[0], config.hidden_dim)),  # 全連接層
            nn.ReLU(),  # ReLU 激活函數
        )

        # 定義輸出層，輸出兩個座標
        self.coord_out = orthogonal_init(nn.Linear(config.hidden_dim * 2, 2))  # hidden_dim*2 -> 2

        # 定義Critic輸出層
        self.critic_out = orthogonal_init(nn.Linear(config.hidden_dim * 2, 1), gain=1.0)  # hidden_dim*2 -> 1

    # 獲取特徵
    def get_feature(self, s_map, s_sensor):
        s_map = s_map.float() / 255.0  # 將地圖數據歸一化
        s_map = self.map_layer(s_map)  # 通過地圖層提取特徵

        s_sensor = self.sensor_layer(s_sensor)  # 通過感測層提取特徵
        feature = torch.cat([s_map, s_sensor], dim=-1)  # 將兩個特徵拼接

        return feature  # 返回拼接后的特徵

    # 獲取座標和價值
    def get_coord_and_value(self, s_map, s_sensor):
        feature = self.get_feature(s_map, s_sensor)  # 獲取特徵
        coord = self.coord_out(feature)  # 通過座標輸出層獲取座標
        value = self.critic_out(feature)  # 通過Critic輸出層獲取價值

        # 將坐標限制在地圖範圍內
        map_height, map_width = s_map.shape[2], s_map.shape[3]
        coord[:, 0] = torch.clamp(coord[:, 0], 0, map_width - 1)
        coord[:, 1] = torch.clamp(coord[:, 1], 0, map_height - 1)

        return coord.int(), value.squeeze(-1)  # 返回整數形式的座標和價值，並移除多余的維度

    # 獲取Critic的輸出
    def critic(self, s_map, s_sensor):
        feature = self.get_feature(s_map, s_sensor)  # 獲取特徵
        value = self.critic_out(feature)  # 通過Critic輸出層獲取價值

        return value.squeeze(-1)  # 返回價值，並移除多余的維度
