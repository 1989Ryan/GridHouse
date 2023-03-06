import gymnasium as gym
from PIL import Image
env = gym.make("MiniGrid-GridHouseEnv-v0", render_mode="rgb_array")
observation, info = env.reset(seed=12)
for _ in range(2):
#    action = policy(observation)  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(2)
    img = env.render()
    img_ = Image.fromarray(img)
    img_.save("rgb.png")
    print(observation['image'].shape)
    print(observation['mission'])

    if terminated or truncated:
        observation, info = env.reset()
env.close()