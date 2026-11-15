from pyqit import datasets
from pyqit.tests.scenarios.base import BaseScenario


class AdHocScenario(BaseScenario):
    def _load(self):
        return datasets.make_ad_hoc(self.n_samples, self.n_features, seed=self.seed)


class LinearlySeparableScenario(BaseScenario):
    def _load(self):
        return datasets.make_linearly_separable(
            self.n_samples, self.n_features, seed=self.seed
        )


class TwoCurvesScenario(BaseScenario):
    def _load(self):
        return datasets.make_two_curves(self.n_samples, self.n_features, seed=self.seed)


class HiddenManifoldScenario(BaseScenario):
    def _load(self):
        return datasets.make_hidden_manifold(
            self.n_samples, self.n_features, seed=self.seed
        )


class HyperplanesParityScenario(BaseScenario):
    def _load(self):
        return datasets.make_hyperplanes_parity(
            self.n_samples, self.n_features, seed=self.seed
        )


class BarsAndStripesScenario(BaseScenario):
    _tags = {"n_features": (4, 9, 16)}

    def _load(self):
        side = round(self.n_features**0.5)
        return datasets.make_bars_and_stripes(
            self.n_samples, height=side, width=side, seed=self.seed
        )


class MoonsScenario(BaseScenario):
    _tags = {"n_features": (2,)}

    def _load(self):
        return datasets.make_moons(self.n_samples, seed=self.seed)


class CirclesScenario(BaseScenario):
    _tags = {"n_features": (2,)}

    def _load(self):
        return datasets.make_circles(self.n_samples, seed=self.seed)
