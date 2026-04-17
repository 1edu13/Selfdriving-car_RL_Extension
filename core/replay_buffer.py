"""
Efficient Experience Replay Buffer for off-policy RL algorithms (DQN, TD3, SAC).

Uses a list with circular indexing instead of collections.deque, providing O(1)
random access for sampling instead of O(n). This makes buffer.sample() dramatically
faster for large replay buffers (100K-500K transitions).

Why not deque?
  - Python deque has O(n) indexed access for elements near the middle.
  - random.sample() accesses random indices → most hits the slow O(n) path.
  - For a 200K buffer with batch_size=256, deque sampling is ~200K times slower
    per index lookup compared to a list.
"""

import numpy as np
import random


class ReplayBuffer:
    """
    Circular buffer that stores transitions and supports fast random sampling.

    Key difference from deque-based buffers:
      - Python list has O(1) indexed access → random.sample() is O(k)
      - Python deque has O(n) indexed access → random.sample() is O(k*n)

    For a buffer of 200K entries and batch_size of 256, this is orders of
    magnitude faster per sample call.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """Store a single transition. Overwrites oldest entry when full."""
        # Force storing states as uint8 to save 75% RAM
        state = np.asarray(state, dtype=np.uint8)
        next_state = np.asarray(next_state, dtype=np.uint8)
        
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a random batch of transitions. Returns numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state, dtype=np.uint8), np.array(action),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.uint8),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
