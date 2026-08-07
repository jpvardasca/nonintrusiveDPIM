import numpy as np
from scipy.sparse import csr_matrix

class VelocityMappingComputer:
    """Computes velocity mappings (Wv) for terms in the DPIM expansion.
    
    This class efficiently computes the velocity mapping terms using:
    - The current displacement solution (Wu)
    - Reduced dynamics coefficients (f)
    - Eigenmodes (Phi)
    - Precomputed μ_γ terms
    """

    def __init__(self, M, Phi, Wu, f, query, Wv=None):
        """
        Initialize the velocity mapping computer.
        
        Args:
            M (sparse matrix): Mass matrix
            Phi (ndarray): Matrix of eigenmodes (columns are modes)
            Wu (ndarray): Matrix of displacement solutions
            f (ndarray): Matrix of reduced dynamics coefficients
            query: Object for querying term combinations
            Wv (ndarray, optional): Existing velocity solutions to update
        """
        self.M = M
        self.Phi = Phi
        self.Wu = Wu
        self.f = f
        self.query = query
        self.Wv = Wv if Wv is not None else np.zeros_like(Wu)
        self.n_modes = Phi.shape[1]

    def compute_velocity_mapping(self, gamma, i_gl, mu_gamma):
        """
        Compute velocity mapping for a specific term using optimized μ_γ.
        
        The velocity mapping is computed as:
            Wv_γ = Wu_γ * s + μ_γ + Φ @ f_γ
            
        Args:
            gamma (ndarray): Exponent vector for current term
            i_gl (int): Global index of current term
            mu_gamma (ndarray): Precomputed μ_γ term
            
        Returns:
            ndarray: Computed velocity mapping Wv_γ
        """
        print(f"\nComputing velocity mapping for gamma = {gamma}, i_gl = {i_gl}")
        s = np.dot(self.lambda_tilde, gamma)
        Wu_gamma = self.Wu[:, i_gl-1]
        f_gamma = self.f[:, i_gl-1]  # Vectorized access
        
        #print(f"s = {s}, Wu_gamma norm = {np.linalg.norm(Wu_gamma)}, f_gamma = {f_gamma}")
        #print(f"Computed mu_gamma norm: {np.linalg.norm(mu_gamma)}")
        
        # Combine terms efficiently
        Wv_gamma = Wu_gamma * s + mu_gamma + self.Phi @ f_gamma
        self.Wv[:, i_gl-1] = Wv_gamma
        
        #print(f"Final Wv_gamma norm: {np.linalg.norm(Wv_gamma)}")
        print(f"Updated Wv for index {i_gl-1}")
        return Wv_gamma


class MuGammaCalculator:
    """Computes the μ_γ term used in velocity mapping calculations.
    
    The μ_γ term represents the cumulative effect of lower-order terms
    on the current term's velocity mapping.
    """

    def __init__(self, M, Phi, Wu, f, query):
        """
        Initialize the μ_γ calculator.
        
        Args:
            M (sparse matrix): Mass matrix
            Phi (ndarray): Matrix of eigenmodes
            Wu (ndarray): Matrix of displacement solutions
            f (ndarray): Matrix of reduced dynamics coefficients
            query: Object for querying term combinations
        """
        self.M = M
        self.Phi = Phi
        self.Wu = Wu
        self.f = f
        self.query = query
        self.n_modes = Phi.shape[1]
    
    def compute_mu_gamma(self, gamma):
        """
        Compute the complete μ_γ term for a given exponent vector.
        
        Args:
            gamma (ndarray): Exponent vector for current term
            
        Returns:
            ndarray: Computed μ_γ term
        """
        #print(f"///////////////////MU GAMMA //////////////////////////////////")
        gamma = np.array(gamma)
        #print(f"gamma:{gamma}")
        mu_gamma = np.zeros(self.M.shape[0], dtype=np.complex128)
        mu_gamma += self._compute_first_term(gamma)
        mu_gamma += self._compute_second_term(gamma)
        return mu_gamma

    def _compute_first_term(self, gamma):
        """
        Compute the first component of μ_γ involving lower-order combinations.
        
        Args:
            gamma (ndarray): Current exponent vector
            
        Returns:
            ndarray: First term contribution to μ_γ
        """
        term = np.zeros(self.M.shape[0], dtype=np.complex128)
        lower_terms = self.query.find_less_combinations(tuple(gamma))
        #print(f"print the lower_terms {lower_terms}")
        if not lower_terms.empty:
            valid_terms = lower_terms[lower_terms['exponents'].map(lambda x: np.sum(x) >= 2)]
            #print(f"print the valid  lower_terms {valid_terms}")
            for _, row in valid_terms.iterrows():
                #print(f"print the valid  lower_terms {row} and gamma {gamma}")
                term += self._process_beta_row_first_term(row, gamma)
        return term

    def _process_beta_row_first_term(self, row, gamma):
        """
        Process a single lower-order term (β) for the first μ_γ component.
        
        Args:
            row: DataFrame row containing β term information
            gamma (ndarray): Current exponent vector
            
        Returns:
            ndarray: Contribution from this β term
        """
        beta = np.array(row['exponents'])
        beta_gl_idx = row['global_index']
        #print(f"beta analysed {beta} and beta_gl_idx {beta_gl_idx}")
        partial_sum = np.zeros(self.M.shape[0], dtype=np.complex128)
        for i in range(self.n_modes):
            exp = gamma - beta
            exp[i] += 1
            term_row = self.query.find_combination(tuple(exp))
            #print(f"term_row {term_row}")
            if term_row is None:
                continue
            term_gl_idx = term_row['global_index'].values[0]
            coeff = gamma[i] - beta[i] + 1
            partial_sum += self.Wu[:, term_gl_idx-1] * self.f[i, beta_gl_idx-1] * coeff
        return partial_sum

    def _compute_second_term(self, gamma):
        """
        Compute the second component of μ_γ involving first-order terms.
        
        Args:
            gamma (ndarray): Current exponent vector
            
        Returns:
            ndarray: Second term contribution to μ_γ
        """
        term = np.zeros(self.M.shape[0], dtype=np.complex128)
        for i in range(self.n_modes):
            term += self._process_mode_i_second_term(i, gamma)
        return term

    def _process_mode_i_second_term(self, i, gamma):
        """
        Process contribution from mode i for the second μ_γ component.
        
        Args:
            i (int): Mode index
            gamma (ndarray): Current exponent vector
            
        Returns:
            ndarray: Contribution from this mode
        """
        e_i_vec = np.zeros(len(gamma), dtype=int)
        e_i_vec[i] = 1
        e_i_tuple = tuple(e_i_vec)
        e_i_row = self.query.find_combination(e_i_tuple)
        if e_i_row is None:
            return 0
        betas = self.query.find_less_combinations_with_order1(e_i_tuple)
        partial_sum = np.zeros(self.M.shape[0], dtype=np.complex128)
        for _, beta_row in betas.iterrows():
            partial_sum += self._process_beta_row_second_term(beta_row, gamma, i)
        return partial_sum

    def _process_beta_row_second_term(self, beta_row, gamma, i):
        """
        Process a single β term for the second μ_γ component.
        
        Args:
            beta_row: DataFrame row containing β term information
            gamma (ndarray): Current exponent vector
            i (int): Current mode index
            
        Returns:
            ndarray: Contribution from this β term
        """
        beta_exp = np.array(beta_row['exponents'])
        beta_gl_idx = beta_row['global_index']
        if not np.all(gamma >= beta_exp):
            return 0
        exp_W = gamma - beta_exp
        exp_W[i] += 1
        term_row = self.query.find_combination(tuple(exp_W))
        if term_row is None:
            return 0
        term_gl_idx = term_row['global_index'].values[0]
        coeff = gamma[i] + 1
        return self.Wu[:, term_gl_idx-1] * self.f[i, beta_gl_idx-1] * coeff