import gymnasium as gym
from PIL import Image
env = gym.make("MiniGrid-GridHouseEnv-v0", render_mode="rgb_array")
observation, info = env.reset(seed=26)
actions = env.gen_traj()
img = env.render()
img_ = Image.fromarray(img)
img_.save("demo/rgb.png")
for i in range(len(actions)):
    observation, reward, terminated, truncated, info = env.step(actions[i])
    img = env.render()
    img_ = Image.fromarray(img)
    img_.save(f"demo/rgb_{i}.png")
    if terminated or truncated:
        observation, info = env.reset()
env.close()