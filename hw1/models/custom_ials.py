"""iALS с таргетом из оценки и давности отзыва. Веса остаются прежними."""
import numpy as np
from .ials import IALS


class CustomIALS(IALS):
    def __init__(self, tau_days=365, floor=.5, **kwargs):
        super().__init__(**kwargs)
        self.tau_days, self.floor = tau_days, floor

    def targets(self, events, cutoff):
        age = np.maximum(0, cutoff - events.timestamp.to_numpy()) / 86400
        recency = self.floor + (1 - self.floor) * np.exp(-age / self.tau_days)
        # До поправки на давность: 1 звезда даёт -1, 3 — ноль, 5 — единицу.
        return ((events.rating.to_numpy() - 3) / 2) * recency
