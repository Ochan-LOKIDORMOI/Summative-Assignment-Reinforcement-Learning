import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from enum import Enum
import random


class WildlifeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    class State(Enum):
        NO_WILDLIFE = 0
        WILDLIFE_DISTANT = 1
        WILDLIFE_APPROACHING = 2
        WILDLIFE_IN_CROP = 3

    class Action(Enum):
        NO_ACTION = 0
        EARLY_WARNING = 1
        DETERRENCE = 2
        EMERGENCY_ALERT = 3

    def __init__(self, render_mode=None):
        super(WildlifeEnv, self).__init__()
        self.grid_size = 5
        self.farm_pos = (2, 2)

        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            "wildlife_pos": spaces.Box(low=0, high=self.grid_size-1, shape=(5, 2), dtype=np.int32),
            "state": spaces.Discrete(len(self.State))
        })

        self.action_space = spaces.Discrete(8)

        # Initialize
        self.agent_pos = None
        self.wildlife = None
        self.current_state = None
        self.steps = 0
        self.max_steps = 150

        # Movement control
        self.last_wildlife_move = 0
        self.wildlife_move_interval = 2.0

        # Rendering
        self.render_mode = render_mode
        self.window_size = 500
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array(self.farm_pos, dtype=np.int32)

        # Initialize 2-4 wildlife at edges
        num_wildlife = self.np_random.integers(2, 5)
        self.wildlife = []

        edges = ['top', 'bottom', 'left', 'right']
        for _ in range(num_wildlife):
            edge = self.np_random.choice(edges)
            if edge == 'top':
                pos = (self.np_random.integers(0, self.grid_size), 0)
            elif edge == 'bottom':
                pos = (self.np_random.integers(
                    0, self.grid_size), self.grid_size-1)
            elif edge == 'left':
                pos = (0, self.np_random.integers(0, self.grid_size))
            else:
                pos = (self.grid_size-1,
                       self.np_random.integers(0, self.grid_size))
            self.wildlife.append(np.array(pos, dtype=np.int32))

        self.current_state = self.State.NO_WILDLIFE
        self.steps = 0
        self.last_wildlife_move = 0
        return self._get_obs(), {}

    def _get_obs(self):
        wildlife_pos = np.zeros((5, 2), dtype=np.int32)
        for i, pos in enumerate(self.wildlife):
            wildlife_pos[i] = pos
        return {
            "agent_pos": self.agent_pos,
            "wildlife_pos": wildlife_pos,
            "state": self.current_state.value
        }

    def _get_distance(self, pos1, pos2):
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def _update_state(self):
        if not self.wildlife:
            self.current_state = self.State.NO_WILDLIFE
            return

        min_distance = min(self._get_distance(self.agent_pos, w)
                           for w in self.wildlife)
        if min_distance <= 1.0:
            self.current_state = self.State.WILDLIFE_IN_CROP
        elif min_distance <= 2.0:
            self.current_state = self.State.WILDLIFE_APPROACHING
        elif min_distance <= 3.5:
            self.current_state = self.State.WILDLIFE_DISTANT
        else:
            self.current_state = self.State.NO_WILDLIFE

    def _move_wildlife(self, current_time):
        if current_time - self.last_wildlife_move < self.wildlife_move_interval:
            return False

        moved = False
        for i, wildlife in enumerate(self.wildlife):
            # 60% chance to move toward farm, 40% random
            if random.random() < 0.6:
                direction = np.sign(self.farm_pos - wildlife)
            else:
                direction = np.array(
                    [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])])

            new_pos = wildlife + direction
            new_pos = np.clip(new_pos, 0, self.grid_size-1)

            if not np.array_equal(new_pos, wildlife):
                self.wildlife[i] = new_pos
                moved = True

        self.last_wildlife_move = current_time
        return moved

    def step(self, action):
        reward = -0.1  # Small penalty for each step
        terminated = False
        truncated = False
        info = {
            'agent_on_wildlife': False,
            'wildlife_on_farm': False
        }
        current_time = time.time()

        # Move agent
        direction = action % 4
        if direction == 0:  # Up
            new_pos = self.agent_pos + np.array([0, -1])
        elif direction == 1:  # Right
            new_pos = self.agent_pos + np.array([1, 0])
        elif direction == 2:  # Down
            new_pos = self.agent_pos + np.array([0, 1])
        else:  # Left
            new_pos = self.agent_pos + np.array([-1, 0])

        new_pos = np.clip(new_pos, 0, self.grid_size-1)
        self.agent_pos = new_pos

        # Check if agent captured wildlife (blue on red)
        for i in range(len(self.wildlife)-1, -1, -1):
            if np.array_equal(self.agent_pos, self.wildlife[i]):
                reward += 5.0  # Large reward for capturing wildlife
                self.wildlife.pop(i)
                info['agent_on_wildlife'] = True

        # Move wildlife and check if any stepped on farm
        if self._move_wildlife(current_time):
            for i in range(len(self.wildlife)-1, -1, -1):
                if np.array_equal(self.wildlife[i], self.farm_pos):
                    reward += -2.0  # Penalty for wildlife on farm
                    info['wildlife_on_farm'] = True

        self._update_state()

        # Termination conditions
        if any(np.array_equal(w, self.farm_pos) for w in self.wildlife):
            terminated = True
            info['outcome'] = 'crop_damage'

        if not self.wildlife:
            terminated = True
            info['outcome'] = 'success'

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
            info['outcome'] = 'timeout'

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == 'human':
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
            pygame.display.set_caption('Wildlife Intrusion Prevention')
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)

        # Draw farm (center cell)
        farm_rect = pygame.Rect(
            self.farm_pos[0] * self.cell_size,
            self.farm_pos[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(canvas, (144, 238, 144), farm_rect)  # Light green

        # Draw agent (blue)
        agent_rect = pygame.Rect(
            self.agent_pos[0] * self.cell_size + self.cell_size // 4,
            self.agent_pos[1] * self.cell_size + self.cell_size // 4,
            self.cell_size // 2,
            self.cell_size // 2
        )
        pygame.draw.rect(canvas, (0, 0, 255), agent_rect)

        # Draw wildlife (red)
        for wildlife in self.wildlife:
            wildlife_rect = pygame.Rect(
                wildlife[0] * self.cell_size + self.cell_size // 4,
                wildlife[1] * self.cell_size + self.cell_size // 4,
                self.cell_size // 2,
                self.cell_size // 2
            )
            pygame.draw.rect(canvas, (255, 0, 0), wildlife_rect)

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
