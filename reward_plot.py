import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("LunarLander-v2-PPO/rewards.npy")
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.grid(True)
plt.show()
