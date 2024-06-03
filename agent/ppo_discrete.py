import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import numpy as np
from .network import Actor_Critic
from .buffer import PPO_Buffer
from utils.data_augmentation import rotation_s

class PPO_Discrete:
    def __init__(self, config, num_envs):
        self.config = config  # 保存配置
        self.device = config.device  # 設置設備（CPU 或 GPU）
        self.minibatches = config.minibatches  # 設置小批次數量
        self.max_train_steps = config.max_train_steps  # 設置最大訓練步數
        self.lr = config.lr  # Actor 的學習率
        self.gamma = config.gamma  # 折扣因子
        self.lamda = config.lamda  # GAE 參數
        self.epsilon = config.epsilon  # PPO 剪裁參數
        self.K_epochs = config.K_epochs  # PPO 參數
        self.entropy_coef = config.entropy_coef  # 熵係數
        self.critic_loss_coef = config.critic_loss_coef  # Critic 損失係數
        self.set_adam_eps = config.set_adam_eps  # 設置 Adam epsilon
        self.use_grad_clip = config.use_grad_clip  # 是否使用梯度剪裁
        self.use_adv_norm = config.use_adv_norm  # 是否使用優勢正則化
        self.use_AdamW = config.use_AdamW  # 是否使用 AdamW 優化器
        self.weight_decay = config.weight_decay  # 權重衰減
        self.use_DrAC = config.use_DrAC  # 是否使用 DrAC
        self.aug_coef = config.aug_coef  # 增強係數

        self.ppo_buffer = PPO_Buffer(config, num_envs)  # 初始化 PPO 緩衝區
        self.ac_net = Actor_Critic(config).to(self.device)  # 初始化 Actor-Critic 網絡並移動到設備

        # 設置優化器
        if self.set_adam_eps:  # 如果設置了 Adam epsilon
            if self.use_AdamW:
                self.optimizer = torch.optim.AdamW(self.ac_net.parameters(), lr=self.lr, eps=1e-5, weight_decay=self.weight_decay)  # 使用 AdamW 優化器
            else:
                self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=self.lr, eps=1e-5)  # 使用 Adam 優化器
        else:
            if self.use_AdamW:
                self.optimizer = torch.optim.AdamW(self.ac_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # 使用 AdamW 優化器
            else:
                self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=self.lr)  # 使用 Adam 優化器

    # 評估策略
    def evaluate_policy(self, s):
        with torch.no_grad():
            s_map = torch.from_numpy(s['s_map']).unsqueeze(0).to(self.device)  # 將地圖狀態轉為張量並移動到設備
            s_sensor = torch.from_numpy(s['s_sensor']).unsqueeze(0).to(self.device)  # 將感測器狀態轉為張量並移動到設備

            logit = self.ac_net.actor(s_map, s_sensor)  # 獲取 actor 輸出
            a = logit.argmax(dim=-1)  # 選擇概率最大的動作
            return a.cpu().numpy()  # 返回動作並移動到 CPU

    # 獲取價值
    def get_value(self, s):
        with torch.no_grad():
            s_map = torch.from_numpy(s['s_map']).to(self.device)  # 將地圖狀態轉為張量並移動到設備
            s_sensor = torch.from_numpy(s['s_sensor']).to(self.device)  # 將感測器狀態轉為張量並移動到設備
            value = self.ac_net.critic(s_map, s_sensor)  # 獲取 critic 輸出
            return value

    # 獲取動作和價值
    def get_action_and_value(self, s):
        with torch.no_grad():
            s_map = torch.from_numpy(s['s_map']).to(self.device)  # 將地圖狀態轉為張量並移動到設備
            s_sensor = torch.from_numpy(s['s_sensor']).to(self.device)  # 將感測器狀態轉為張量並移動到設備

            logit, value = self.ac_net.get_logit_and_value(s_map, s_sensor)  # 獲取 logit 和價值
            a = logit.argmax(dim=-1)  # 選擇概率最大的動作
            logprob = F.log_softmax(logit, dim=-1).gather(1, a.unsqueeze(-1)).squeeze(-1)  # 獲取動作的對數概率
            return a.cpu().numpy(), logprob, value, {'s_map': s_map, 's_sensor': s_sensor}  # 返回動作、對數概率、價值和狀態

    # 更新策略
    def update(self):
        batch = self.ppo_buffer.get_training_data()  # 獲取訓練數據
        batch_size = batch['a'].shape[0]  # 訓練數據大小
        minibatch_size = batch_size // self.minibatches  # 小批次大小

        # 優化策略 K 輪
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, False):
                logits_now, values_now = self.ac_net.get_logit_and_value(batch['s_map'][index], batch['s_sensor'][index])  # 獲取當前 logit 和價值
                dist_now = Categorical(logits=logits_now)  # 創建當前的類別分佈
                dist_entropy = dist_now.entropy()  # 計算熵
                entropy_loss = dist_entropy.mean()  # 熵損失

                logprob_now = dist_now.log_prob(batch['a'][index])  # 獲取當前的對數概率
                # 比率 = exp(log(a) - log(b))
                ratios = torch.exp(logprob_now - batch['logprob'][index])  # 計算比率
                surr1 = ratios * batch['adv'][index]  # 計算 surr1
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['adv'][index]  # 計算 surr2
                actor_loss = -torch.min(surr1, surr2).mean()  # Actor 損失

                critic_loss = 0.5 * F.mse_loss(batch['v_target'][index], values_now)  # Critic 損失

                ppo_loss = actor_loss - self.entropy_coef * entropy_loss + self.critic_loss_coef * critic_loss  # PPO 損失

                if self.use_DrAC:  # 如果使用 DrAC
                    aug_s_map, aug_s_sensor = rotation_s(s_map=batch['s_map'][index], s_sensor=batch['s_sensor'][index], k=np.random.randint(1, 4))  # 增強數據
                    aug_logits, aug_values = self.ac_net.get_logit_and_value(aug_s_map, aug_s_sensor)  # 獲取增強后的 logit 和價值

                    aug_actor_loss = F.kl_div(input=torch.log_softmax(aug_logits, dim=-1), target=torch.softmax(logits_now, dim=-1).detach(), reduction='batchmean')  # 增強 Actor 損失
                    aug_critic_loss = 0.5 * F.mse_loss(values_now.detach(), aug_values)  # 增強 Critic 損失

                    loss = ppo_loss + (aug_actor_loss + aug_critic_loss) * self.aug_coef  # 總損失
                else:
                    loss = ppo_loss  # 總損失

                self.optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 損失反向傳播
                if self.use_grad_clip:  # 如果使用梯度剪裁
                    torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)  # 梯度剪裁
                self.optimizer.step()  # 優化器步驟

    # 重置緩衝區
    def reset_buffer(self, num_envs):
        self.ppo_buffer = PPO_Buffer(config=self.config, num_envs=num_envs)  # 重置 PPO 緩衝區

    # 保存模型
    def save_model(self, algorithm, env_version, map_name, number, seed, index):
        torch.save(self.ac_net.state_dict(), "./model/{}_env_{}_{}_number_{}_seed_{}_index_{}.pth".format(algorithm, env_version, map_name, number, seed, index))  # 保存模型狀態字典
