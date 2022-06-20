import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import scipy
    import numpy as np
    from cyanure import estimators
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'Cyanure'

    install_cmd = 'conda'
    requirements = ['cyanure']
    references = [
        'J. Mairal, "Cyanure: An Open-Source Toolbox for Empirical Risk'
        ' Minimization for Python, C++, and soon more," '
        'Arxiv eprint 1912.08165 (2019)'
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        if (scipy.sparse.issparse(self.X) and
                scipy.sparse.isspmatrix_csc(self.X)):
            self.X = scipy.sparse.csr_matrix(self.X)

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        n_samples = self.X.shape[0]

        self.solver_parameter = dict(
            lambda_1=self.lmbd / n_samples, solver='auto', duality_gap_interval=1000000,
            tol=1e-12
        )
        self.solver = estimators.Lasso(fit_intercept=fit_intercept, verbose=False, **self.solver_parameter)
        

    def run(self, n_iter):
        self.solver.max_iter = n_iter
        self.solver.fit(self.X, self.y)

    def get_result(self):
        beta = self.solver.get_weights()
        if self.fit_intercept:
            beta, intercept = beta
            beta = np.r_[beta.flatten(), intercept]
        return beta
