import torch
import random


# Train Genertor
class Environment:
    # The number of selectable actions
    N_WORD = 10

    # The Shape of the brush
    _brush = {
        0: [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]],
        1: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
        2: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
        3: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
        4: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
        5: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
        6: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
        7: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
        8: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
        9: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
    }

    # Table of correspondences between action and movement
    _move_table = {
        0: (0, 0),
        1: (0, 0),
        2: (0, 1),
        3: (1, 1),
        4: (1, 0),
        5: (1, -1),
        6: (0, -1),
        7: (-1, -1),
        8: (-1, 0),
        9: (-1, 1),
    }

    def __init__(self, D, subject_img, img_size, stroke_length, random_start=True):
        self.D = D
        self.stroke_length = stroke_length
        self.stroke_current = 0

        # image dimension
        self.img_size = img_size
        self.px0 = img_size // 2
        self.py0 = img_size // 2

        bound = img_size // 2 - 1
        if random_start:
            # -31 <= dx <= 30 : 1 <= x <= 62
            self.dx = random.randint(-bound, bound - 1)
            self.dy = random.randint(-bound, bound - 1)
        else:
            self.dx = 0
            self.dy = 0

        noise = torch.randn_like(subject_img) * 0.1
        subject_img = subject_img + noise
        # (1, 1, W, H) -> (1, W, H)
        self.subject_img = torch.clamp(subject_img, max=1.0, min=0.0).squeeze(0)

        # (1, W, H)
        self.canvas = torch.zeros(img_size, img_size, requires_grad=False).unsqueeze(0)

        self.first_reward = True
        self.last_reward = 0
        self.initialized = False

    def _act(self, canvas, word, pos, image_size):
        depth = -1
        (dx, dy) = self._move_table[word]

        rx, ry = pos[0], pos[1]

        # Check if the cursor position is inside of the canvas.
        # Because Brush size is 3x3, range check is 1 <= current pos < size - 1
        is_inner = lambda pos, delta, size: 1 <= pos + delta < size - 1

        # Check if the brush position is inside or not for all dimension
        tfs = [is_inner(pos, delta, image_size) for pos, delta in zip((rx, ry), (dx, dy))]
        if all(tfs):
            rx += dx
            ry += dy

            dotimg = torch.tensor(self._brush[word]).to('cpu')
            canvas[rx - 1:rx + 1 + 1, ry - 1:ry + 1 + 1] += dotimg * depth
            canvas = torch.clamp(canvas, max=0, min=-1)

        return rx, ry, canvas

    def _canvas_to_img(self, canvas):
        return canvas + 1

    def _roll_and_fill(self, img, x, y):
        moved_img = torch.roll(img, shifts=(x, y), dims=(1, 2))
        if x > 0:
            moved_img[:, :x, :] = 1
        elif x < 0:
            moved_img[:, x:, :] = 1
        if y > 0:
            moved_img[:, :, :y] = 1
        elif y < 0:
            moved_img[:, :, y:] = 1

        return moved_img

    def _observation(self):
        img = self._canvas_to_img(self.canvas)

        st1 = torch.nn.functional.pad(img, (
            self.img_size // 2,
            self.img_size // 2,
            self.img_size // 2,
            self.img_size // 2),
            mode='constant', value=1)

        st2 = torch.nn.functional.pad(self.subject_img, (
            self.img_size // 2,
            self.img_size // 2,
            self.img_size // 2,
            self.img_size // 2),
            mode='constant', value=1)

        # (1, 2*width, 2*height) -> (1, 2*width, 2*height)
        st1 = self._roll_and_fill(st1, -self.dx, -self.dy)
        # (1, 2*width, 2*height) -> (1, 2*width, 2*height)
        st2 = self._roll_and_fill(st2, -self.dx, -self.dy)

        # (1, W, H) x (1, W, H) -> (2, W, H)
        return torch.cat((st1, st2), dim=0).detach()

    def first_state(self):
        return self._observation()

    def get_subject_img(self):
        return self.subject_img

    def get_img(self):
        return self._canvas_to_img(self.canvas.detach())

    def step(self, action):
        px = self.px0 + self.dx
        py = self.py0 + self.dy

        canvas = self.canvas.squeeze(0)
        px, py, canvas = self._act(canvas, action, (px, py), self.img_size)
        self.canvas = canvas.unsqueeze(0)

        self.dx = px - self.px0
        self.dy = py - self.py0

        img = self._canvas_to_img(self.canvas)

        # (1, width, height) -> (1, 1, width, height)
        di1 = img.unsqueeze(0).detach()
        # (1, width, height) -> (1, 1, width, height)
        di2 = self.subject_img.unsqueeze(0).detach()

        # Calcurate reward
        reward_swap = self.D(di1, di2)
        reward = 0 if self.first_reward else reward_swap[0] - self.last_reward
        self.last_reward = reward_swap[0]
        self.first_reward = False

        # Clip reward
        if reward < -0.0001:
            reward = -0.000001
        elif reward < 0:
            reward = 0
        elif reward > 0 and reward < 0.0001:
            reward = 0.0001
        elif reward >= 0.0001:
            reward = 0.0002

        self.stroke_current += 1
        done = 1 if self.stroke_current >= self.stroke_length else 0

        return self._observation(), float(reward), done
