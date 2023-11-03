import numpy as np
from scipy import stats

class MixtureModel(stats.rv_continuous):
    def __init__(self, submodels, *args, weights = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
        
    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x)  * weight
        return pdf
            
    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x)  * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x)  * weight
        return cdf

        

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs

uniform = stats.uniform(loc=0, scale = 1)
low_t_heavy = MixtureModel([
                              stats.uniform(loc=0, scale = 0.025),
                              stats.uniform(loc=0, scale = 1)],
                             weights = [0.25, 0.75])