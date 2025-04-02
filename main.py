import os
import time
import random
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import WildlifeEnv
import pygame
from pygame.locals import *

# Configuration
MODEL_TYPE = "ppo"  # "dqn" or "ppo"
USE_MODEL = True
NUM_EPISODES = 10
RENDER_MODE = "human"


class SimulationRunner:
    def __init__(self):
        self.model_path = f"models/{MODEL_TYPE}/{MODEL_TYPE}_wildlife_final.zip"
        self.algorithm = MODEL_TYPE
        self.render_mode = RENDER_MODE

        # Initialize environment with the same settings as training
        self.base_env = WildlifeEnv(render_mode="rgb_array")
        self.env = DummyVecEnv([lambda: Monitor(self.base_env)])

        if USE_MODEL and os.path.exists(self.model_path):
            print(
                f"Loading {self.algorithm.upper()} model from: {self.model_path}")
            if self.algorithm == 'dqn':
                self.model = DQN.load(self.model_path, env=self.env)
            else:
                self.model = PPO.load(self.model_path, env=self.env)
        else:
            print("Running with random actions")
            self.model = None

        # Movement control (aligned with environment)
        self.agent_move_interval = 0.3
        self.wildlife_move_interval = 2.0
        self.last_agent_move = time.time()
        self.last_wildlife_move = time.time()

        # Pygame setup
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            pygame.display.set_caption(
                'Wildlife Intrusion Prevention Simulation')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 18)

        # Statistics
        self.total_reward = 0.0
        self.episode_count = 0
        self.steps = 0
        self.episode_rewards = []
        self.wildlife_captured = 0
        self.farm_penalties = 0

    def run_episode(self):
        obs = self.env.reset()
        done = [False]
        episode_reward = 0.0
        exploration_rate = 0.2 if self.model else 1.0

        while not done[0]:
            current_time = time.time()

            if self.render_mode == 'human':
                for event in pygame.event.get():
                    if event.type == QUIT:
                        return False

            # Agent movement
            if current_time - self.last_agent_move >= self.agent_move_interval:
                exploration_rate = max(0.01, exploration_rate * 0.995)

                if self.model and random.random() > exploration_rate:
                    action, _ = self.model.predict(obs, deterministic=False)
                else:
                    action = np.array([random.randint(0, 7)])

                obs, reward, done, info = self.env.step(action)

                # Track custom rewards
                if info[0].get('agent_on_wildlife', False):
                    self.wildlife_captured += 1
                if info[0].get('wildlife_on_farm', False):
                    self.farm_penalties += 1

                episode_reward += float(reward[0])
                self.steps += 1
                self.last_agent_move = current_time

                if self.render_mode == 'human':
                    self.render(episode_reward, exploration_rate)
                    self.clock.tick(10)

            # Wildlife movement
            if current_time - self.last_wildlife_move >= self.wildlife_move_interval:
                obs, _, _, _ = self.env.step(np.array([0]))  # NOOP
                self.last_wildlife_move = current_time

            if done[0]:
                self.total_reward += episode_reward
                self.episode_count += 1
                self.episode_rewards.append(episode_reward)
                outcome = info[0].get('outcome', 'unknown')
                print(
                    f"Episode {self.episode_count}: Reward {episode_reward:.1f}, Outcome: {outcome}")
                print(
                    f"  Wildlife captured: {self.wildlife_captured}, Farm penalties: {self.farm_penalties}")
                time.sleep(1)
                return True

        return True

    def render(self, current_reward, exploration_rate):
        self.screen.fill((255, 255, 255))
        frame = self.base_env.render()

        if frame is not None:
            frame_surface = pygame.surfarray.make_surface(frame)
            frame_surface = pygame.transform.scale(frame_surface, (600, 600))
            self.screen.blit(frame_surface, (0, 0))

        stats = [
            f"Episode: {self.episode_count + 1}",
            f"Step: {self.steps}",
            f"Current Reward: {current_reward:.1f}",
            f"Total Reward: {self.total_reward:.1f}",
            f"Wildlife Captured: {self.wildlife_captured}",
            f"Farm Penalties: {self.farm_penalties}",
            f"Algorithm: {self.algorithm.upper()}",
            f"Exploration: {exploration_rate:.1%}" if self.model else "Random Actions"
        ]

        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (0, 0, 0))
            self.screen.blit(text, (10, 10 + i * 25))

        pygame.display.flip()

    def run(self):
        print(f"\n{'='*50}")
        print(
            f"Starting simulation with {self.algorithm.upper() if USE_MODEL else 'random'} actions")
        print(f"Number of episodes: {NUM_EPISODES}")
        print(f"Reward System:")
        print(f"  +5.0 for capturing wildlife (blue on red)")
        print(f"  -2.0 for wildlife on farm (red on green)")
        print(f"  -0.1 per step")
        print(f"{'='*50}\n")

        for _ in range(NUM_EPISODES):
            if not self.run_episode():
                break

        if self.episode_count > 0:
            print(f"\n{'='*50}")
            print(f"Simulation completed")
            print(f"Total episodes: {self.episode_count}")
            print(f"Total steps: {self.steps}")
            print(f"Total wildlife captured: {self.wildlife_captured}")
            print(f"Total farm penalties: {self.farm_penalties}")
            print(f"Average reward: {np.mean(self.episode_rewards):.1f}")
            print(
                f"Success rate: {sum(1 for r in self.episode_rewards if r > 0)/self.episode_count:.1%}")
            print(f"{'='*50}")

        if self.render_mode == 'human':
            pygame.quit()


if __name__ == "__main__":
    runner = SimulationRunner()
    runner.run()
