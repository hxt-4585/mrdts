"""随机 Fourier 特征和递推岭回归；分别预测物理成本，不学习队列加权奖励。

每次更新 O(p² + po)，每批 K 个候选预测 O(K(dp + p² + po))。
不确定性是回归杠杆值乘历史残差的启发式尺度，不是校准后的置信区间。
"""

import numpy as np
import torch


class OnlineSurrogate:
    def __init__(self, inputs, outputs, *, features=32, ridge=1., seed=45):
        rng = np.random.default_rng(seed)
        self.frequencies = rng.normal(0., 1. / np.sqrt(inputs), (inputs, features))
        self.phases = rng.uniform(0., 2. * np.pi, features)
        width = 1 + inputs + features
        self.precision_inverse = np.eye(width) / ridge
        self.weights = np.zeros((width, outputs))
        self.residual_squares = np.zeros(outputs)
        self.samples = 0

    def transform(self, inputs):
        inputs = np.atleast_2d(inputs)
        random = np.sqrt(2. / len(self.phases)) * np.cos(inputs @ self.frequencies + self.phases)
        return np.column_stack((np.ones(len(inputs)), inputs, random))

    def predict(self, inputs):
        basis = self.transform(inputs)
        leverage = np.einsum('ki,ij,kj->k', basis, self.precision_inverse, basis)
        scale = np.sqrt(np.maximum(self.residual_squares / max(self.samples, 1), 1e-6))
        return basis @ self.weights, np.sqrt(np.maximum(leverage, 0.))[:, None] * scale

    def update(self, inputs, target):
        basis = self.transform(inputs)[0]
        residual = target - basis @ self.weights
        projected = self.precision_inverse @ basis
        gain = projected / (1. + basis @ projected)
        self.weights += np.outer(gain, residual)
        self.precision_inverse -= np.outer(gain, projected)
        self.residual_squares += residual ** 2
        self.samples += 1
        return residual

    def state(self):
        return dict(samples=self.samples, **{key: torch.from_numpy(getattr(self, key).copy())
                    for key in ('frequencies', 'phases', 'precision_inverse', 'weights', 'residual_squares')})

    @classmethod
    def from_state(cls, state):
        model = cls.__new__(cls)
        model.samples = state['samples']
        for key in ('frequencies', 'phases', 'precision_inverse', 'weights', 'residual_squares'):
            setattr(model, key, state[key].cpu().numpy().copy())
        return model
