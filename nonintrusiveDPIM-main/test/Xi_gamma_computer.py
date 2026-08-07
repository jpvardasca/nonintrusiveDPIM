import numpy as np
from scipy.sparse import csr_matrix

class XiGammaComputer:
    def __init__(self, M, C, Wu, Wv, f, query, lambda_tilde, n_modes):
        """Initialize xi_gamma computer with system matrices and solution data.
        
        Args:
            M: Mass matrix (sparse)
            C: Damping matrix (sparse)
            Wu: Nonlinear displacement coefficients
            Wv: Nonlinear velocity coefficients
            f: Reduced dynamics coefficients
            query: Exponent table query interface
            lambda_tilde: Extended eigenvalue vector
            n_modes: Number of modal coordinates
        """
        # Convert to sparse matrices for efficient computation
        self.M = csr_matrix(M)
        self.C = csr_matrix(C)
        # Solution matrices
        self.Wu = Wu
        self.Wv = Wv
        self.f = f
        # Query interface for exponent table
        self.query = query
        # Extended eigenvalues (modes + forcing)
        self.lambda_tilde = lambda_tilde
        # Number of modal coordinates
        self.n_modes = n_modes
        
    def compute_xi_gamma(self, gamma, i_gl):
        """Compute the xi_gamma vector for a given exponent vector gamma.
        
        Args:
            gamma: Exponent vector (polynomial term)
            i_gl: Global index of the term
            
        Returns:
            xi_gamma: Computed vector for normal form analysis
        """

        print(f"\nComputing xi_gamma for gamma = {gamma}, i_gl = {i_gl}")
        # Initialize result vector
        xi_gamma_initial = np.zeros(self.Wu.shape[0], dtype=np.complex128)
        xi_gamma = np.zeros(self.Wu.shape[0], dtype=np.complex128)
        # Compute eigenvalue combination: s = γ·λ̃
        s = np.dot(self.lambda_tilde, gamma)
        
        # Find lower-order terms (β < γ)
        lower_terms = self.query.find_less_combinations(gamma)
        if not lower_terms.empty:
            print(f"Found {len(lower_terms)} lower terms (first part)")
            # Process each lower-order term
            for _, row in lower_terms.iterrows():
                xi_gamma = self._process_beta_row(row, gamma, s, xi_gamma_initial) + xi_gamma
        
        # Add contributions from order-1 forcing terms
        xi_gamma = self._add_second_part_contribution(gamma, xi_gamma_initial, s) + xi_gamma

        return -xi_gamma
        
    def _add_second_part_contribution(self, gamma, xi_gamma, s):
        """Add contributions from forcing unit vectors (second part of equation).
        
        Args:
            gamma: Current exponent vector
            xi_gamma: Current xi_gamma vector
            s: Dot product γ·λ̃
            
        Returns:
            Updated xi_gamma vector
        """
            
        print("Processing second part (order-1 forcing unit vectors)")
        n_total = len(gamma)
        
        # Iterate through modal directions
        for i in range(self.n_modes):
            # Create unit vector in mode-i direction
            e_i_vec = np.zeros(n_total, dtype=int)
            e_i_vec[i] = 1
            e_i_tuple = tuple(e_i_vec)
            
            # Find corresponding row in exponent table
            e_i_row = self.query.find_combination(e_i_tuple)

            if e_i_row is None:
                print(f"  Mode {i}: unit vector not found, skipping")
                continue
                
            # Find β terms with β ≤ e_i
            beta_df = self.query.find_less_combinations_with_order1(e_i_vec)

            if beta_df.empty:
                print(f"  Mode {i}: No valid beta terms found")
                continue
                
            print(f"  Mode {i}: Found {len(beta_df)} beta terms")
            # Process each β term
            for _, beta_row in beta_df.iterrows():

                beta = np.array(beta_row['exponents'])
                beta_global_idx = beta_row['global_index']
                # Identify forcing variable position
                j = np.argmax(beta)  

                # Skip if gamma can't accommodate this β
                if gamma[j] < 1:  
                    continue

                # Compute new exponent: γ - β + e_i
                exp = gamma.copy()

                exp[j] -= 1  # Remove forcing unit
                exp[i] += 1   # Add mode unit
                term = self.query.find_combination(tuple(exp))

                
                if term is None:
                    continue
                    
                # Get solution data for the new term
                term_global_idx = term['global_index'].values[0]
                f_beta_i = self.f[i, beta_global_idx-1]
                coeff = gamma[i] + 1  # Coefficient from derivative
                Wu_term = self.Wu[:, term_global_idx-1]
                Wv_term = self.Wv[:, term_global_idx-1]
                
                # Compute contribution using sparse operations
                term_contribution = (
                    (s * self.M + self.C) @ Wu_term + 
                    self.M @ Wv_term
                ) * f_beta_i * coeff
                
                # Add to cumulative result
                xi_gamma += term_contribution
                print(f"    Added contribution from beta={beta} (norm={np.linalg.norm(term_contribution)})")
                
        return xi_gamma
        
    def _process_beta_row(self, row, gamma, s, xi_gamma):
        """Process contributions from a single β term.
        
        Args:
            row: DataFrame row with β term information
            gamma: Current exponent vector
            s: Dot product γ·λ̃
            xi_gamma: Current xi_gamma vector
        """
        beta = np.array(row['exponents'])
        beta_global_idx = row['global_index']
        print(f"\nProcessing lower term beta = {beta}, global index {beta_global_idx}")
        
        # Iterate through modal directions
        for i in range(self.n_modes):
            # Check if gamma has sufficient exponent in mode-i
            if gamma[i] >= beta[i]:
                term_contribution = self._compute_term_contribution(
                    beta, gamma, beta_global_idx, i, s
                )
                # Subtract contribution (negative sign in equation)
                xi_gamma = term_contribution + xi_gamma
        return xi_gamma
                
    def _compute_term_contribution(self, beta, gamma, beta_global_idx, i, s):
        """Compute contribution for a specific (β, mode-i) combination.
        
        Args:
            beta: Lower-order exponent vector
            gamma: Current exponent vector
            beta_global_idx: Global index of β term
            i: Mode index
            s: Dot product γ·λ̃
            
        Returns:
            Contribution vector to xi_gamma
        """
        # Compute new exponent: γ - β + e_i
        exp = gamma - beta
        exp[i] += 1
        print(f"  Checking mode {i+1}: looking for exp = {exp}")
        term = self.query.find_combination(exp)
        
        if term is None:
            print("    Term not found")
            return 0.0  # Zero contribution if term not present
            
        # Get solution data for the new term
        term_global_idx = term['global_index'].values[0]
        coeff = gamma[i] - beta[i] + 1  # Coefficient from derivative
        f_beta_i = self.f[i, beta_global_idx-1]
        Wu_term = self.Wu[:, term_global_idx-1]
        Wv_term = self.Wv[:, term_global_idx-1]
        
        # Print debug information
        self._print_term_info(term_global_idx, coeff, f_beta_i, Wu_term, Wv_term)
        # Compute actual vector contribution
        term_contribution = self._compute_vector_contribution(
            s, Wu_term, Wv_term, f_beta_i, coeff
        )

        return term_contribution
        
    def _print_term_info(self, term_global_idx, coeff, f_beta_i, Wu_term, Wv_term):
        """Print detailed information about a term contribution.
        
        Args:
            term_global_idx: Global index of term
            coeff: Combinatorial coefficient
            f_beta_i: Reduced dynamics coefficient
            Wu_term: Displacement solution vector
            Wv_term: Velocity solution vector
        """
        print(f"    Found term at global index {term_global_idx}")
        print(f"    coeff = {coeff}, f_beta_i = {f_beta_i}")
        print(f"    Wu_term norm = {np.linalg.norm(Wu_term)}")
        print(f"    Wv_term norm = {np.linalg.norm(Wv_term)}")
        
    def _compute_vector_contribution(self, s, Wu_term, Wv_term, f_beta_i, coeff):
        """Compute the vector contribution term.
        
        Args:
            s: Dot product γ·λ̃
            Wu_term: Displacement solution vector
            Wv_term: Velocity solution vector
            f_beta_i: Reduced dynamics coefficient
            coeff: Combinatorial coefficient
            
        Returns:
            Computed vector contribution
        """
        # Convert to dense for small operations (sparse would be inefficient)
        C_dense = self.C.toarray()
        M_dense = self.M.toarray()
        # Compute: [(s*M + C) @ Wu_term + M @ Wv_term] * f_beta_i * coeff
        return (
            ((s * M_dense + C_dense) @ Wu_term + M_dense @ Wv_term) 
            * f_beta_i * coeff
        )