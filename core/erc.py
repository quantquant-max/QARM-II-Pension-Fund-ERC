import numpy as np

def erc_weights(Sigma: np.ndarray) -> np.ndarray:
    """
    Placeholder: renvoie des poids égaux (on branchera le solveur ERC réel ensuite).
    """
    n = Sigma.shape[0]
    if n == 0 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a non-empty square matrix.")
    return np.ones(n) / n
