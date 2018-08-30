from numpy.linalg import multi_dot, pinv

from .loo_utils import loo, loo_kern_inv


class CovModifier(object):
    def fit(self, X, y):
        return self

    def update(self):
        return self

    def inv_dot(self):
        raise NotImplemented('This function must be implemented in a subclass')

    def loo_inv_dot(self):
        raise NotImplemented('This function must be implemented in a subclass')

    def get_x0(self):
        return None

    def get_bounds(self):
        return None

    def copy(self):
        raise NotImplemented('This function must be implemented in a subclass')


class ShrinkageModifier(CovModifier):
    def __init__(self, alpha=0.5):
        if alpha < 1E-15:
            raise ValueError('alpha must be greater than zero')
        self.alpha = alpha

    def fit(self, X, y):
        self.A_inv = 1 / float(self.alpha)
        self.B = (1 - self.alpha) * X.T
        # self.B = X.T
        self.G = self.A_inv * self.B
        self.K = X.dot(self.G)
        self.K.flat[::len(self.K) + 1] += 1
        return self

    def update(self, alpha):
        return ShrinkageModifier(alpha)

    def inv_dot(self, X, P):
        """Computes inv(cov(X)) @ P"""
        A_inv_P = self.A_inv * P
        return A_inv_P - multi_dot((self.G, pinv(self.K), X, A_inv_P))

    def loo_inv_dot(self, X, Ps):
        """Computes inv(cov(X)) @ P in a leave-one-out scheme"""
        Xs = loo(X)
        Gs = loo(self.G, axis=1)
        K_invs = loo_kern_inv(self.K)

        for G, K_inv, X, P in zip(Gs, K_invs, Xs, Ps):
            A_inv_P = self.A_inv * P
            yield A_inv_P - multi_dot((G, K_inv, X, A_inv_P))

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0.1, 1)]

    def copy(self):
        s = ShrinkageModifier(self.alpha)
        s.A_inv = self.A_inv
        s.B = self.B
        s.G = self.G
        s.K = self.K
        return s
