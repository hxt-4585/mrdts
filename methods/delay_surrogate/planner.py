"""复用原基线的局部特征、候选与协商，仅学习和评价区域时延。"""

import numpy as np

from methods.local_surrogate.planner import LocalPlanner


class DelayPlanner(LocalPlanner):
    def __init__(self, env, *, candidates=32, rounds=2, features=32, ridge=1.,
                 seed=45, uncertainty_penalty=.05):
        super().__init__(env, candidates=candidates, rounds=rounds, features=features,
                         ridge=ridge, seed=seed, uncertainty_penalty=uncertainty_penalty)
        # 保留同 seed 的随机特征及回归协方差，仅保留时延输出，不训练能耗预测器。
        for model in self.models:
            model.weights = model.weights[:, :1].copy()
            model.residual_squares = model.residual_squares[:1].copy()

    def cost(self, owner, view, proposals):
        # 共享特征不包含 Q；评分既不读取 Q，也不调用飞行能耗模型。
        prediction, uncertainty = self.models[owner].predict(view.features(proposals))
        return np.maximum(prediction[:, 0], 0.) + self.uncertainty_penalty * uncertainty[:, 0]
