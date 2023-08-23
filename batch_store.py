from collections import namedtuple, deque
import numpy as np


ExpElem = namedtuple('ExpElem', ['state', 'action', 'reward', 'q'])
Experience = namedtuple('Experience', ['key', 'state', 'action', 'reward', 'next_state', 'done', 'q', 'next_q'])


class BatchStore:
    def __init__(self, name, n_steps, gamma):
        self.name = str(name)
        self.experience_deque = deque(maxlen=n_steps)
        self.experiences = []
        self.n_steps = n_steps
        self.gamma = gamma
        self.batch_num = 0

    def _multi_step_reward(self, experiences, reward):
        ret = 0.
        for idx, exp in enumerate(experiences):
            ret += exp.reward * (self.gamma ** idx)
        ret += reward * (self.gamma ** len(experiences))
        return ret

    def add(self, state, action, reward, done, qs):
        if len(self.experience_deque) == self.n_steps or done:
            key = self.name + '-' + str(self.batch_num)
            t0_state, t0_action, _, t0_qs = self.experience_deque[0]
            t0_reward = self._multi_step_reward(self.experience_deque, reward)
            tp_n_state = state
            tp_n_qs = qs
            tp_done = np.float32(done)
            experience = Experience(key,
                                    t0_state,
                                    t0_action,
                                    t0_reward,
                                    tp_n_state,
                                    tp_done,
                                    t0_qs,
                                    tp_n_qs)
            self.experiences.append(experience)
            self.batch_num = self.batch_num + 1

        if done:
            self.experience_deque.clear()

        if not done:
            self.experience_deque.append(ExpElem(state, action, reward, qs))

    def reset(self):
        self.experiences = []
        self.experience_deque.clear()

    def _get_priorities(self):
        actions = np.array([experience.action for experience in self.experiences], dtype=np.int32)
        rewards = np.array([experience.reward for experience in self.experiences])
        dones = np.array([experience.done for experience in self.experiences])

        qs = np.stack([experience.q for experience in self.experiences])
        nqs = np.stack([experience.next_q for experience in self.experiences])

        qs = qs[(range(len(qs)), actions)]
        next_qs = np.max(nqs, axis=1)

        expected_qs = rewards + (self.gamma ** self.n_steps) * next_qs * (1 - dones)
        td_error = expected_qs - qs
        prios = np.abs(td_error) + 1e-6
        return prios

    def make_batch(self):
        prios = self._get_priorities()
        batch = Experience(*zip(*self.experiences))
        return batch, prios

    def __len__(self):
        return len(self.experiences)


if __name__ == "__main__":
    print("test")
    # Test data
    states = [1, 2, 3, 4, 5]
    actions = [0, 1, 2, 3, 4]
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
    dones = [False, False, False, False, True]
    qs = [[0.5, 0.3, 0.2, 0.1, 0.4],
          [0.4, 0.2, 0.3, 0.1, 0.5],
          [0.3, 0.5, 0.4, 0.2, 0.1],
          [0.2, 0.1, 0.3, 0.5, 0.4],
          [0.1, 0.2, 0.3, 0.4, 0.5]]

    # batch store
    batch_store = BatchStore('test', n_steps=1, gamma=0.9)

    # add data
    for state, action, reward, done, q in zip(states, actions, rewards, dones, qs):
        batch_store.add(state, action, reward, done, q)

    # make batch
    batch, priorities = batch_store.make_batch()

    # show batch data
    print('Batch Data:')

    for elem in zip(*batch):
        experience = Experience(*elem)
        print(f'Key: {experience.key}')
        print(f'State: {experience.state}')
        print(f'Action: {experience.action}')
        print(f'Reward: {experience.reward}')
        print(f'Next State: {experience.next_state}')
        print(f'Done: {experience.done}')
        print(f'Q: {experience.q}')
        print(f'Next Q: {experience.next_q}')
        print()

    # show priorities
    print(f'Priorities: {priorities}')

    # reset
    batch_store.reset()

    # zero
    print(f'Batch Store Length: {len(batch_store)}')
