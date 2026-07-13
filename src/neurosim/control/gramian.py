
import numpy as np
import scipy as sp
from numpy.linalg import eig
from numpy import matmul as mm, transpose as tp


def compute_gramian(A_norm, T, B=None, system=None):
    
    if system is None:
        raise Exception(
            "Time system not specified. "
            "Please nominate whether you are using a continuous-time or a discrete-time system."
        )
    elif system != "continuous" and system != "discrete":
        raise Exception(
            "Incorrect system specification. "
            "Please specify either 'system=discrete' or 'system=continuous'."
        )

    n_nodes = A_norm.shape[0]

    if B is None:
        B = np.eye(n_nodes)

    w, _ = eig(A_norm)
    BB = mm(B, tp(B))


    if T == np.inf:
        if system == "continuous":
            if np.max(np.real(w)) < 0:
                return sp.linalg.solve_continuous_lyapunov(A_norm, -BB)
            else:
                raise Exception(
                    "Cannot compute infinite-time Gramian for an unstable continuous-time system. "
                    "Ensure max(real(eigenvalues(A_norm))) < 0 before calling compute_gramian(T=np.inf)."
                )
        elif system == "discrete":
            if np.max(np.abs(w)) < 1:
                return sp.linalg.solve_discrete_lyapunov(A_norm, BB)
            else:
                raise Exception(
                    "Cannot compute infinite-time Gramian for an unstable discrete-time system. "
                    "Ensure spectral_radius(A_norm) < 1 before calling compute_gramian(T=np.inf)."
                )

    if system == "continuous":
       
        STEP = 0.001
        t = np.arange(0, T + STEP / 2, STEP)
        n_steps = len(t)

        dE = sp.linalg.expm(A_norm * STEP)
        
        Wc = mm(B, B.T)  # Initial term, Simpson weight = 1
        dEa_running = np.eye(n_nodes)

        for i in range(1, n_steps - 1):
            dEa_running = mm(dEa_running, dE)
            dEab = mm(dEa_running, B)
            dG_curr = mm(dEab, dEab.T)
            weight = 4.0 if (i % 2 != 0) else 2.0
            Wc += weight * dG_curr

        if n_steps > 1:
            dEa_running = mm(dEa_running, dE)
            dEab = mm(dEa_running, B)
            dG_curr = mm(dEab, dEab.T)
            Wc += dG_curr

        Wc *= (STEP / 3.0)

        return Wc

    elif system == "discrete":
       
        T = int(T)
        Ap = np.eye(n_nodes)
        Wc = mm(B, tp(B))
        for _ in range(T):
            Ap = mm(Ap, A_norm)
            Wc = Wc + mm(mm(Ap, BB), tp(Ap))
        return Wc



