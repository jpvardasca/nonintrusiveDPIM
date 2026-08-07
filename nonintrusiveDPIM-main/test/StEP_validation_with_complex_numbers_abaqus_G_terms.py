import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from Compute_all_options_K_tg import TangentStiffnessMatrixHandler
from Exponent_Query import ExponentQuery

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from Compute_all_options_K_tg import TangentStiffnessMatrixHandler
from Exponent_Query import ExponentQuery

class ComputeGterms:
    def __init__(self, K, W_u_dict, non_embedded_nodes, base_input_filename, exponent_generator, n_modes, n_forcing, minimum_distance):
        """
        Initialize the G terms computer with necessary components
        """
        self.K = K
        self.W_u = W_u_dict
        self.non_embedded_nodes = non_embedded_nodes
        self.exponent_generator=exponent_generator.exponent_table
        self.base_input_filename = base_input_filename
        self.n_modes=n_modes
        self.n_forcing=n_forcing
        self.minimum_distance=minimum_distance
        self.exponent_query = ExponentQuery(
            exponent_generator.exponent_table,
            exponent_generator.order_table,
            exponent_generator.exponent_dict,
            self.n_modes,
            self.n_forcing
        )
        self.boxminus_cache = {}  # Cache for K_boxminus matrices
        self.tangent_cache = {}
        self.sign_cache = {}   # Cache for tangent stiffness matrices

        # Corrected: Use exponent_generator instead of undefined 'generator'
        self.finder = QuadraticCombinationFinder(
            exponent_table=exponent_generator.exponent_table,
            order_table=exponent_generator.order_table,
            exponent_dict=exponent_generator.exponent_dict
        )

    def compute_G(self, gamma):
        """
        Compute the G coefficient for a specific monomial gamma
        """
        print(f"\nComputing G for gamma: {gamma}")
        G = 0
        
        # Corrected: Handle symmetric pairs properly
        symmetric_pairs = self.finder.find_symmetric_combinations(gamma)
        print(f"symmetric pair: {symmetric_pairs}")
        if symmetric_pairs:
            sigma = symmetric_pairs[0]['sigma']  # Get sigma from the pair
            print(f"Processing symmetric case with sigma: {sigma}")
            symmetric_term = self._compute_symmetric_term(sigma)
            print(f"Symmetric term contribution: {symmetric_term}")
            G += symmetric_term
        
        # Corrected: Iterate through asymmetric pairs
        asymmetric_pairs = self.finder.find_asymmetric_combinations(gamma)
        print(f"assymmetric pair: {asymmetric_pairs}")
        
        for i, pair in enumerate(asymmetric_pairs, 1):
            sigma = pair['sigma']
            nu = pair['nu']
            print(f"  Pair {i}: sigma={sigma}, nu={nu}")
    
            asymmetric_term = self._compute_asymmetric_term(sigma, nu)
            print(f"Pair ({sigma}, {nu}) contribution: {asymmetric_term}")
            G += asymmetric_term
            
        print(f"Total G for gamma {gamma}: {G}")
        return G



    def _get_K_tg(self, exponent, part, sign):
        """Compute tangent stiffness matrix with caching"""
        key = (exponent, part, sign)
        if key in self.tangent_cache:
            return self.tangent_cache[key]
        
        # Get displacement vector
        W_sigma = self.W_u[exponent]
        disp_vector = W_sigma.real if part == 'real' else W_sigma.imag
        scale_factor=(0.24*self.minimum_distance/(np.max(np.abs(disp_vector))) if np.any(disp_vector != 0) 
                          else  1)
        sign_mixed=1

        # If displacement is zero, return linear stiffness matrix K
        if np.all(disp_vector == 0):
            self.tangent_cache[key] = self.K
            self.sign_cache[key]=sign_mixed
            return self.K
        
        # Otherwise compute tangent stiffness
        handler = TangentStiffnessMatrixHandler(
            option=1,
            base_input_filename=self.base_input_filename,
            non_embedded_nodes=self.non_embedded_nodes,
            exponent_table=self.exponent_generator,
            exponent=exponent,
            part=part,
            sign=sign,
            displacement=W_sigma.flatten().tolist(),
            minimum_distance=self.minimum_distance
        )
        K_tg= handler.compute_or_load()
        self.tangent_cache[key] = K_tg
        self.sign_cache[key]=sign_mixed
        return K_tg, scale_factor

    def _get_K_boxminus(self, exponent, part):
        """Compute K_boxminus for a given exponent and part (real/imag)"""
        key = (exponent, part)
        if key in self.boxminus_cache:
            return self.boxminus_cache[key]
        
        # Get displacement vector
        W_sigma = self.W_u[exponent]

        
        # Get tangent stiffness matrices
        K_pos, scale_factor = self._get_K_tg(exponent, part, 'pos')
        K_neg, scale_factor = self._get_K_tg(exponent, part, 'neg')
        
        # Compute K_boxminus as difference
        K_boxminus = K_pos - K_neg
        self.boxminus_cache[key] = K_boxminus
        return K_boxminus, scale_factor

    def _compute_symmetric_term(self, sigma):
        """Compute G term when σ = ν = γ/2"""
        print("\n" + "-"*50)
        print(f"COMPUTING SYMMETRIC TERM for sigma: {sigma}")
        print("-"*50)
        W_sigma = self.W_u[sigma]
        
        K_boxminus_real, scale_factor_r = (self._get_K_boxminus(sigma, 'real') 
                          if np.any(self.W_u[sigma].real != 0) 
                          else (csr_matrix(self.K.shape) , 1))
        #print(scale_factor_r)
        K_boxminus_imag, scale_factor_i = (self._get_K_boxminus(sigma, 'imag') 
                          if np.any(self.W_u[sigma].imag != 0) 
                          else (csr_matrix(self.K.shape), 1))
        

        print(f"G symmetric scale factor r and i: {scale_factor_r, scale_factor_i}")
        #print(K_boxminus_real @ W_sigma.real)
        # Compute the three components
        G_rr = (K_boxminus_real @ W_sigma.real) / (4*scale_factor_r)
        G_ii = (K_boxminus_imag @ W_sigma.imag) / (4*scale_factor_i)
        G_ri = (K_boxminus_real @ W_sigma.imag) / (2*scale_factor_r)
        
        # Combine components according to formula
        return G_rr - G_ii + 1j * G_ri

    def _compute_asymmetric_term(self, sigma, nu):
        """Compute G term for σ ≠ ν"""
        print("\n" + "-"*50)
        print(f"COMPUTING ASYMMETRIC TERM for sigma={sigma}, nu={nu}")
        print("-"*50)
        W_nu = self.W_u[nu]
        
        # Only compute K_boxminus if displacement is non-zero
        K_boxminus_real, scale_factor_r = (self._get_K_boxminus(sigma, 'real') 
                          if np.any(self.W_u[sigma].real != 0) 
                          else ( csr_matrix(self.K.shape), 1))
        K_boxminus_imag, scale_factor_i = (self._get_K_boxminus(sigma, 'imag') 
                          if np.any(self.W_u[sigma].imag != 0) 
                          else (csr_matrix(self.K.shape), 1))

        #print(f"K shape: {K_boxminus_imag}")
        print(f"G assymmetric scale factor r and i: {scale_factor_r, scale_factor_i}")
        # Compute the four components
        G_rr = (K_boxminus_real @ W_nu.real) / (2*scale_factor_r)
        G_ii = (K_boxminus_imag @ W_nu.imag) / (2*scale_factor_i)
        G_ri = (K_boxminus_real @ W_nu.imag) / (2*scale_factor_r)
        G_ir = (K_boxminus_imag @ W_nu.real) / (2*scale_factor_i)
        
        # Combine components according to formula
        return (G_rr - G_ii) + 1j * (G_ri + G_ir)




class QuadraticCombinationFinder:
    def __init__(self, exponent_table, order_table, exponent_dict):
        """
        Initialize with exponent tables and dictionary from MultiExponentGenerator.
        
        Args:
            exponent_table: DataFrame containing exponents and their global indices
            order_table: DataFrame containing order information
            exponent_dict: Dictionary mapping exponent tuples to indices
        """
        self.exponent_table = exponent_table
        self.order_table = order_table
        self.exponent_dict = exponent_dict

    #def find_quadratic_combinations(self, gamma):
        #"""Find all quadratic combinations (sigma, nu) such that sigma + nu = gamma."""
        #symmetric = self.find_symmetric_combinations(gamma)
        #asymmetric = self.find_asymmetric_combinations(gamma)
        #return symmetric + asymmetric

    def find_symmetric_combinations(self, gamma):
        """Find symmetric pairs where sigma = nu = gamma/2."""
        gamma_array = np.array(gamma)
        # Check if all components are even
        if not np.all(gamma_array % 2 == 0):
            return []
        
        half_gamma = tuple(gamma_array // 2)
        # Check if half-exponent exists in our dictionary
        if half_gamma in self.exponent_dict:
            return [{'sigma': half_gamma, 'nu': half_gamma}]
        return []

    def find_asymmetric_combinations(self, gamma):
        """Find asymmetric pairs where sigma < nu (by global index)."""
        gamma_order = sum(gamma)
        gamma_array = np.array(gamma)
        combinations = []

        # Get indices of exponents with order less than gamma's order
        lower_order_indices = self.order_table[
            self.order_table['order'] < gamma_order
        ]
        
        if lower_order_indices.empty:
            return combinations

        # Get the range of global indices for lower-order exponents
        first_idx = lower_order_indices['first_index'].min()
        last_idx = lower_order_indices['last_index'].max()
        
        # Retrieve candidate exponents within the order range
        candidates = self.exponent_table[
            (self.exponent_table['global_index'] >= first_idx) & 
            (self.exponent_table['global_index'] <= last_idx)
        ]
        
        # Filter candidates where sigma <= gamma (component-wise)
        sigma_candidates = candidates[
            np.all(np.array(list(candidates['exponents'])) <= gamma_array, axis=1)
        ]
        
        # Find valid asymmetric pairs
        for _, row in sigma_candidates.iterrows():
            sigma = row['exponents']
            sigma_idx = row['global_index']
            nu = tuple(gamma_array - np.array(sigma))
            
            # Skip symmetric case (handled separately)
            if sigma == nu:
                continue
                
            # Check if nu exists in exponent dictionary
            if nu in self.exponent_dict:
                nu_idx = self.exponent_dict[nu] + 1  # Convert to 1-indexed
                
                # Ensure sigma has smaller index to avoid duplicates
                if sigma_idx <= nu_idx:
                    combinations.append({'sigma': sigma, 'nu': nu})
        
        return combinations



