import numpy as np


class LinearOperatorCache:
    """Caches and manages linear operators used in the dynamic power iteration method.
    
    This class efficiently computes and caches linear operators that combine:
    - The system matrices (M, C, K)
    - The eigenmode matrix (Phi)
    - The eigenvalue vector (lambda_)
    for different complex frequencies (s).
    """

    def __init__(self, M, C, K, Phi, lambda_, n_modes):
        """
        Initialize the linear operator cache.
        
        Args:
            M (sparse matrix): Mass matrix
            C (sparse matrix): Damping matrix
            K (sparse matrix): Stiffness matrix
            Phi (ndarray): Matrix of eigenmodes (columns are modes)
            lambda_ (ndarray): Vector of eigenvalues
            n_modes (int): Number of modes in the system
        """
        self.M = M
        self.C = C
        self.K = K
        self.Phi = Phi
        self.lambda_ = lambda_
        self.cache = {}  # Dictionary to cache computed operators
        self.n_modes = n_modes
        
    def _compute_LW(self, s):
        """
        Compute the linear operator L_W(s) = s²M + sC + K.
        
        Args:
            s (complex): Complex frequency parameter
            
        Returns:
            ndarray: Dense matrix of L_W(s)
        """
        return (s**2 * self.M + s * self.C + self.K).toarray()
        
    def _compute_Lf(self, s):
        """
        Compute the linear operator L_f(s) = M·Φ·(sI + Λ) + C·Φ.
        
        Args:
            s (complex): Complex frequency parameter
            
        Returns:
            ndarray: Dense matrix of L_f(s)
        """
        Lambda_matrix = np.diag(self.lambda_)
        sI_plus_Lambda = s * np.eye(self.n_modes) + Lambda_matrix
        
        # Compute L_f = M·Φ·(sI + Λ) + C·Φ
        term1 = self.M.dot(self.Phi.dot(sI_plus_Lambda))
        term2 = self.C.dot(self.Phi)
        L_f = term1 + term2
        return np.asarray(L_f, dtype=np.complex128)
        
    def _combine_operators(self, L_W, L_f):
        """
        Combine L_W and L_f operators into a single block matrix [L_W | L_f].
        
        Args:
            L_W (ndarray): Operator for the W terms
            L_f (ndarray): Operator for the f terms
            
        Returns:
            ndarray: Combined operator matrix
        """
        n, m = L_W.shape[0], L_f.shape[1]
        combined = np.zeros((n, n + m), dtype=np.complex128)
        combined[:, :n] = L_W
        combined[:, n:] = L_f
        return combined
        
    def get_operator(self, s):
        """
        Get the combined linear operator for a given frequency s.
        
        If the operator is not cached, it will be computed and stored.
        
        Args:
            s (complex): Complex frequency parameter
            
        Returns:
            ndarray: Combined operator [L_W(s) | L_f(s)]
        """
        if s in self.cache:
            return self.cache[s]
            
        L_W = self._compute_LW(s)
        L_f = self._compute_Lf(s)
        combined = self._combine_operators(L_W, L_f)
        self.cache[s] = combined
        return combined