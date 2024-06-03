import torch

class PPO_Buffer:
    def __init__(self, config, num_envs):
        self.device = config.device  # 設置設備（CPU 或 GPU）
        self.gamma = config.gamma  # 折扣因子
        self.lamda = config.lamda  # GAE 的 lambda 參數
        self.s_map_dim = config.s_map_dim  # 地圖狀態維度
        self.s_sensor_dim = config.s_sensor_dim  # 感測器狀態維度
        self.rollout_steps = config.rollout_steps  # 回合步數
        self.num_envs = num_envs  # 環境數量
        self.use_adv_norm = config.use_adv_norm  # 是否使用優勢正則化

        # 初始化緩衝區
        self.buffer = {
            's_map': torch.zeros(((self.rollout_steps, self.num_envs,) + self.s_map_dim), dtype=torch.uint8, device=self.device),  # 地圖狀態緩衝區
            's_sensor': torch.zeros(((self.rollout_steps, self.num_envs,) + self.s_sensor_dim), dtype=torch.float32, device=self.device),  # 感測器狀態緩衝區
            'value': torch.zeros((self.rollout_steps + 1, self.num_envs), dtype=torch.float32, device=self.device),  # 價值緩衝區
            'a': torch.zeros((self.rollout_steps, self.num_envs, 2), dtype=torch.int, device=self.device),  # 動作緩衝區，兩個座標
            'logprob': torch.zeros((self.rollout_steps, self.num_envs, 2), dtype=torch.float32, device=self.device),  # 動作對數概率緩衝區，兩個座標的對數概率
            'r': torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device),  # 獎勵緩衝區
            'terminal': torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device),  # 終止標誌緩衝區
        }
        self.count = 0  # 當前緩衝區計數

    # 存儲過渡
    def store_transition(self, s, value, a, logprob, r, terminal):
        self.buffer['s_map'][self.count] = s['s_map']  # 存儲地圖狀態
        self.buffer['s_sensor'][self.count] = s['s_sensor']  # 存儲感測器狀態
        self.buffer['value'][self.count] = value  # 存儲價值
        self.buffer['a'][self.count] = a  # 存儲動作（兩個座標）
        self.buffer['logprob'][self.count] = logprob  # 存儲動作對數概率（兩個座標的對數概率）
        self.buffer['r'][self.count] = torch.tensor(r, dtype=torch.float32, device=self.device)  # 存儲獎勵
        self.buffer['terminal'][self.count] = torch.tensor(terminal, dtype=torch.float32, device=self.device)  # 存儲終止標誌
        self.count += 1  # 更新緩衝區計數

    # 存儲價值
    def store_value(self, value):
        self.buffer['value'][self.count] = value  # 存儲價值

    # 使用 GAE 計算優勢和目標價值
    def get_adv(self):
        with torch.no_grad():  # 不計算梯度
            # self.buffer['value'][:-1]=v(s)
            # self.buffer['value'][1:]=v(s')
            deltas = self.buffer['r'] + self.gamma * (1.0 - self.buffer['terminal']) * self.buffer['value'][1:] - self.buffer['value'][:-1]  # 計算優勢增量
            adv = torch.zeros_like(self.buffer['r'], device=self.device)  # 初始化優勢緩衝區
            gae = 0  # 初始化 GAE
            for t in reversed(range(self.rollout_steps)):  # 從後向前計算
                gae = deltas[t] + self.gamma * self.lamda * gae * (1.0 - self.buffer['terminal'][t])  # 計算 GAE
                adv[t] = gae  # 儲存優勢
            v_target = adv + self.buffer['value'][:-1]  # 計算目標價值
            if self.use_adv_norm:  # 如果使用優勢正則化
                adv = ((adv - torch.mean(adv)) / (torch.std(adv) + 1e-5))  # 正則化優勢

        return adv, v_target  # 返回優勢和目標價值

    # 獲取訓練數據
    def get_training_data(self):
        adv, v_target = self.get_adv()  # 計算優勢和目標價值
        # batch_size = rollout_steps * num_envs
        batch = {
            's_map': self.buffer['s_map'].reshape((-1,) + self.s_map_dim),  # 調整地圖狀態形狀
            's_sensor': self.buffer['s_sensor'].reshape((-1,) + self.s_sensor_dim),  # 調整感測器狀態形狀
            'a': self.buffer['a'].reshape(-1, 2),  # 調整動作形狀（兩個座標）
            'logprob': self.buffer['logprob'].reshape(-1, 2),  # 調整動作對數概率形狀（兩個座標的對數概率）
            'adv': adv.reshape(-1),  # 調整優勢形狀
            'v_target': v_target.reshape(-1),  # 調整目標價值形狀
        }
        return batch  # 返回訓練批次
