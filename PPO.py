import gymnasium as gym
import torch
import numpy as np
import os
import random
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=39):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# PPO是确定性策略梯度法
# actor-critic 结构，有两个网络
class Policy(nn.Module):
    def __init__(self, d_observation, d_action, hid):
        super(Policy, self).__init__()
        self.action = nn.Sequential(    # 策略网络，对策略建模，策略梯度法的体现
            nn.Linear(d_observation, hid),
            nn.ReLU(),
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Linear(64, d_action),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(    # 价值网络，用神经网络对状态价值函数 V建模，在基线技术中用于表示 TD目标
            nn.Linear(d_observation, hid),
            nn.ReLU(),
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

# 经验回放
class Memory:
    def __init__(self):
        self.observations:list = []  # 状态信息
        self.actions:list = []       
        self.rewards:list = []        
        self.terminated:list = []    # 标志是否结束
        self.old_logprob:list = []   # 策略梯度法中的策略概率logP
    def clear(self):
        del self.observations[:]
        del self.actions[:]
        del self.rewards[:]
        del self.terminated[:]
        del self.old_logprob[:]
    def append(self, observation, action, reward, terminated):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminated.append(terminated)
    def append_logprob(self, old_logprob):
        self.old_logprob.append(old_logprob)   # 记录采样时策略的动作概率值

    def Observation(self):
        return torch.tensor(np.array(self.observations), dtype=torch.float32).to(device)
    def Action(self):
        return torch.tensor(self.actions, dtype=torch.int64).to(device)
    def Reward(self):
        return torch.tensor(self.rewards, dtype=torch.float32).to(device)
    def Terminated(self):
        return torch.tensor(self.terminated, dtype=torch.float32).to(device)
    def Old_logprob(self):
        return torch.tensor(self.old_logprob, dtype=torch.float32).to(device)

class Agent:
    def __init__(self, device, s_d, a_d, gamma=0.99, episilon=0.2):
        self.gamma = gamma
        self.policy = Policy(s_d, a_d, 128).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.memory = Memory()
        self.episilon = episilon

    def act(self, observation: torch.Tensor) -> torch.Tensor:   # 根据当前观测选择动作，并记录动作概率对数（用于策略梯度更新）
        prob = self.policy.action(observation)    # 计算策略动作概率值
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()   # 选择动作
        logprob = dist.log_prob(action).squeeze()  
        self.memory.append_logprob(logprob)
        return action.item()

    def evaluate(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor]:   # 评估一批观测和动作
        prob = self.policy.action(observation)
        dist = torch.distributions.Categorical(prob)
        logprob = dist.log_prob(action).squeeze()  
        value = self.policy.critic(observation).squeeze()    # 计算状态价值
        return logprob, value
        
    def update(self, k_epoch=5):
        def normalize(x: torch.Tensor, episilon=1e-5) -> torch.Tensor:
            return (x - x.mean())/(x.std()+episilon)
        
        def discount(self, rewards: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
            discount_reward = []
            r_tmp = 0
            for i in range(rewards.shape[0]-1, -1, -1):   # 从后往前计算奖励，增量式计算
                if terminated[i] == 1:
                    r_tmp = 0
                r_tmp = rewards[i] + self.gamma * r_tmp
                discount_reward.insert(0, r_tmp)
            return torch.stack(discount_reward)
        
        discount_reward = discount(self, self.memory.Reward(), self.memory.Terminated())
        discount_reward = normalize(discount_reward)
        old_logprob = self.memory.Old_logprob()

        mean_loss = 0
        for _ in range(k_epoch):
            logprob, value = self.evaluate(self.memory.Observation(), self.memory.Action())
            advantage = discount_reward - value   # 基线技术
            advantage = normalize(advantage)
            ratio = torch.exp(logprob - old_logprob.detach())    # 利用当前动作策略和采样时的旧策略值，计算PPO重要性采样比
            surr1 = ratio * advantage   # 策略梯度项
            surr2 = torch.clamp(ratio, 1-self.episilon, 1+self.episilon) * advantage   # PPO clip 梯度裁剪，使策略变化更加稳定
            loss = -torch.min(surr1, surr2) + nn.MSELoss()(value, discount_reward)   # PPO损失函数：策略损失+价值损失
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_loss += loss.item()
            
        self.memory.clear()
        return mean_loss/k_epoch

    def append(self, *kwargs):
        self.memory.append(*kwargs)

    def save(self, model_name):
        torch.save(self.policy.state_dict(), model_name)

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", continuous = False, gravity = -10.0,
            enable_wind = False, wind_power=15.0, turbulence_power=1.5, render_mode=None)  # LunarLander-v3 with Gymnasium
    env.reset(seed=39)
    seed_everything()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Agent(device, state_dim, action_dim)
    Num_round = 1500
    episode_length = 1000
    episode_rewards = []
    for episode in range(Num_round):    # 模型训练次数
        total_reward = 0
        observation, info = env.reset()
        for t in range(episode_length):  # 采集数据
            action = agent.act(torch.tensor(observation)[None,:].to(device))
            new_observation, reward, terminated, truncated, info = env.step(action)
            agent.append(observation, action, reward, terminated)
            total_reward += reward
            observation = new_observation
            if terminated:
                observation, info = env.reset()
                break            
        episode_rewards.append(total_reward)
        if episode % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Avg Reward: {avg:.2f}")
        agent.update()   # 模型更新

    agent.save("model_3.pth")
    np.save("rewards_3.npy", episode_rewards)
    env.close()
