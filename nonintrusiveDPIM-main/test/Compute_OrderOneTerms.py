import numpy as np
from scipy.sparse.linalg import spsolve
from collections import defaultdict

class OrderOneTerms:
    """Solves first-order terms in the dynamic power iteration method expansion.
    
    This class handles three types of first-order terms:
    1. Modal terms (eigenmodes of the system)
    2. Forcing terms (external excitation vectors)
    3. Negative forcing terms (complex conjugates of forcing terms)
    """

    def __init__(self, M, C, K, Phi, Psi, lambda_, lambda_tilde, force_vectors, 
                 n_modes, n_forcing, resonance_tol):
        """
        Initialize the first-order terms solver.
        
        Args:
            M (sparse matrix): Mass matrix
            C (sparse matrix): Damping matrix
            K (sparse matrix): Stiffness matrix
            Phi (ndarray): Matrix of right eigenvectors (columns are modes)
            Psi (ndarray): Matrix of left eigenvectors
            lambda_ (ndarray): Vector of eigenvalues
            lambda_tilde (ndarray): Modified eigenvalues
            force_vectors (ndarray): Matrix of external forcing vectors
            n_modes (int): Number of modes in the system
            n_forcing (int): Number of forcing vectors
            resonance_tol (float): Tolerance for resonance detection
        """
        self.M = M
        self.C = C
        self.K = K
        self.Phi = Phi
        self.Psi = Psi
        self.lambda_ = lambda_
        self.lambda_tilde = lambda_tilde
        self.force_vectors = force_vectors
        self.n_modes = n_modes
        self.n_forcing = n_forcing
        self.resonance_tol = resonance_tol
        self.n_dof = M.shape[0]  # Number of degrees of freedom
        self.n_total = n_modes + 2 * n_forcing  # Total number of first-order terms
        
        # Initialize solution arrays
        self.Wu = [None] * self.n_total  # Displacement solutions
        self.Wv = [None] * self.n_total  # Velocity solutions
        self.f = [None] * self.n_total   # Reduced dynamics coefficients

    def solve(self):
        """
        Solve all first-order terms.
        
        Returns:
            tuple: (Wu, Wv, f, res_relations, res_terms) where:
                Wu: List of displacement solutions
                Wv: List of velocity solutions
                f: List of reduced dynamics coefficients
                res_relations: Dictionary mapping resonant modes to their forcing terms
                res_terms: Set of resonant forcing terms
        """
        res_relations = defaultdict(list)  # Tracks resonance relationships
        res_terms = set()                 # Tracks all resonant terms

        for j in range(self.n_total):
            if j < self.n_modes:
                # Solve for modal terms
                self._solve_mode(j)

            elif j < self.n_modes + self.n_forcing:
                # Solve for forcing terms
                k = j - self.n_modes
                force_vec = self.force_vectors[:, k]
                s = self.lambda_tilde[j]
                res_modes = self._solve_forcing(force_vec, s, j)
                if res_modes:
                    # Record resonance relationships
                    exp = tuple(1 if i == j else 0 for i in range(self.n_total))
                    for mode in res_modes:
                        res_relations[mode].append(exp)
                    res_terms.add(exp)
            else:
                # Solve for negative forcing terms (complex conjugates)
                k = j - (self.n_modes + self.n_forcing)
                self._solve_negative_forcing(k, j)
                
        return self.Wu, self.Wv, self.f, res_relations, res_terms

    def _solve_mode(self, j):
        """
        Solve for a modal term (eigenmode solution).
        
        Args:
            j (int): Mode index
        """
        self.Wu[j] = self.Phi[:, j]
        self.Wv[j] = self.lambda_tilde[j] * self.Phi[:, j]
        self.f[j] = np.zeros(self.n_modes, dtype=np.complex128)
        self.f[j][j] = self.lambda_tilde[j]

    def _solve_forcing(self, force_vector, s, j):
        """
        Solve for a forcing term, handling potential resonances.
        
        Args:
            force_vector (ndarray): External forcing vector
            s (complex): Frequency parameter
            j (int): Term index
            
        Returns:
            list: Indices of resonant modes (empty if no resonance)
        """
        res_modes = self._detect_resonances(s)
        if not res_modes:
            # Non-resonant case - simple linear solve
            A = (s**2) * self.M + s * self.C + self.K
            A_csc = A.tocsc()
            self.Wu[j] = spsolve(A_csc, force_vector)
            self.Wv[j] = s * self.Wu[j]
            self.f[j] = np.zeros(self.n_modes, dtype=np.complex128)
            return res_modes

        # Resonant case - augmented system solve
        n_R = len(res_modes)
        Phi_R = self.Phi[:, res_modes]
        Psi_R_H = self.Psi[:, res_modes].conj().T

        Lambda_R = np.diag(self.lambda_[res_modes])
        # Construct augmented system blocks
        term1 = self.M @ Phi_R @ (s * np.eye(n_R) + Lambda_R)
        term2 = self.C @ Phi_R
        B_block = term1 + term2
        term3 = (s * np.eye(n_R) + Lambda_R) @ Psi_R_H @ self.M
        term4 = Psi_R_H @ self.C
        C_block = term3 + term4
        
        D_block = Psi_R_H @ self.M @ Phi_R

        A_block = (s**2) * self.M + s * self.C + self.K
        top = np.hstack([A_block.toarray(), B_block])
        bottom = np.hstack([C_block, D_block])
        full_matrix = np.vstack([top, bottom])

        # Solve augmented system
        rhs = np.concatenate([force_vector, np.zeros(n_R)])
        sol = np.linalg.solve(full_matrix, rhs)
        self.Wu[j] = sol[:self.n_dof]
        f_res = sol[self.n_dof:(self.n_dof + n_R)]
        f_vec = np.zeros(self.n_modes, dtype=np.complex128)
        f_vec[res_modes] = f_res
        self.Wv[j] = s * self.Wu[j]
        self.f[j] = f_vec
        return res_modes

    def _solve_negative_forcing(self, k, j):
        """
        Solve for negative forcing term using conjugate symmetry.
        
        Args:
            k (int): Forcing vector index
            j (int): Term index
        """
        pos_idx = self.n_modes + k
        self.Wu[j] = np.conj(self.Wu[pos_idx])
        self.Wv[j] = np.conj(self.Wv[pos_idx])
        
        # Handle reduced dynamics with permutation for second-order systems
        half = self.n_modes // 2
        f_pos = self.f[pos_idx]
        f_swapped = np.concatenate([f_pos[half:], f_pos[:half]])
        self.f[j] = np.conj(f_swapped)

    def _detect_resonances(self, s):
        """
        Detect modes resonant with the given frequency.
        
        Args:
            s (complex): Frequency parameter
            
        Returns:
            list: Indices of resonant modes
        """
        return [i for i, lam in enumerate(self.lambda_) 
                if abs(s.imag - lam.imag) < self.resonance_tol]