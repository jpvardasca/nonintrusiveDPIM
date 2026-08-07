import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from Compute_all_options_K_tg import TangentStiffnessMatrixHandler
from Exponent_Query import ExponentQuery


class ComputeHterms:
    def __init__(self, K, W_u_dict, non_embedded_nodes, base_input_filename, exponent_generator, n_modes, n_forcing, minimum_distance):
        self.K = K
        self.W_u_dict = W_u_dict
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
        # Caches for efficient reuse
        self.boxplus_cache = {}
        self.tangent_cache = {}
        self.sign_cache = {}
        self.sign_mixed_cache = {}
        self.sign_mixed_pair_cache = {}
        self.mixed_cache = {}
        self.mixed_pair_cache = {}
        self.term_count = 0
        self.nonzero_count = 0

        self.finder = CubicCombinationFinder(
            exponent_table=exponent_generator.exponent_table,
            order_table=exponent_generator.order_table,
            exponent_dict=exponent_generator.exponent_dict
        )

    def compute_H(self, gamma):
        """Compute H coefficient for target exponent gamma of order 3"""
        print(f"\n[DEBUG] Computing H for gamma: {gamma}")
        H_gamma = 0
        symmetric_pairs = self.finder.case1(gamma)
        print(f"symmetric pair: {symmetric_pairs}")


        if symmetric_pairs:
            sigma = symmetric_pairs[0]['sigma']  # Get sigma from the pair
            print(f"Processing symmetric case with sigma: {sigma}")
            symmetric_term = self._case_all_equal(sigma)
            print(f"Symmetric term contribution: {symmetric_term}")
            H_gamma += symmetric_term

        # Corrected: Iterate through asymmetric pairs
        asymmetric_pairs_two_equal = self.finder.case2_and_3(gamma)
        print(f"assymmetric pair: {asymmetric_pairs_two_equal}")
        
        for i, pair in enumerate(asymmetric_pairs_two_equal, 1):
            sigma = pair['sigma']
            kappa = pair['kappa']
            print(f"  Pair {i}: sigma={sigma}, kappa={kappa}")
    
            asymmetric_term = self._case_two_equal(sigma, kappa)
            print(f"Pair ({sigma}, {kappa}) contribution: {asymmetric_term}")
            H_gamma += asymmetric_term

        # Corrected: Iterate through asymmetric pairs
        asymmetric_pairs_all_different = self.finder.case4(gamma)
        print(f"assymmetric pair: {asymmetric_pairs_all_different}")
        
        for i, pair in enumerate(asymmetric_pairs_all_different, 1):
            sigma = pair['sigma']
            nu = pair['nu']
            kappa = pair['kappa']
            print(f"  Pair {i}: sigma={sigma}, nu={nu}, kappa={kappa}")
    
            asymmetric_term_all_dif = self._case_all_distinct(sigma, nu, kappa)
            print(f"Pair ({sigma}, {nu}, {kappa}) contribution: {asymmetric_term_all_dif}")
            H_gamma += asymmetric_term_all_dif
                    
        print(f"Total H for gamma {gamma}: {H_gamma}")
        return H_gamma


    def _case_all_equal(self, sigma):
        """Case 1: All three exponents equal (sigma = nu = kappa)"""
        print(f"    [CASE1] σ={sigma}")
        W_sigma = self.W_u_dict[sigma]
        K_box_r, scale_factor_r = self._get_K_boxplus(sigma, 'real')

        K_box_i, scale_factor_i = self._get_K_boxplus(sigma, 'imag')

        # Handle zero displacement cases
        real_zero = np.all(W_sigma.real == 0)
        imag_zero = np.all(W_sigma.imag == 0)
        if real_zero and imag_zero:
            return 0

        print(f"H all equal scale factor r and i: {scale_factor_r, scale_factor_i}")

        # Compute components
        H_rrr = K_box_r @ W_sigma.real / (6*scale_factor_r**2) if not real_zero else 0
        #print(f"H_rrr: {H_rrr}")
        H_rii = K_box_i @ W_sigma.real / (2*scale_factor_i**2) if not real_zero else 0
        #print(f"H_rii: {H_rii}")
        H_rri = K_box_r @ W_sigma.imag / (2*scale_factor_r**2) if not imag_zero else 0
        #print(f"H_rri: {H_rri}")
        H_iii = K_box_i @ W_sigma.imag / (6*scale_factor_i**2) if not imag_zero else 0
        #print(f"H_iii: {H_iii}")

        return (H_rrr - H_rii) + 1j * (H_rri - H_iii)

    def _case_two_equal(self, sigma, kappa):
        """Case 2: Two exponents equal, one different (sigma, sigma, kappa)"""
        print(f"    [CASE2] σ={sigma}, κ={kappa}")
        W_sigma = self.W_u_dict[sigma]
        W_kappa = self.W_u_dict[kappa]

        mixed_mat, scale_factor_r, scale_factor_i, sign= self._get_K_mixed(sigma)
        K_box_r, scale_factor_r = self._get_K_boxplus(sigma, 'real')
        K_box_i, scale_factor_i = self._get_K_boxplus(sigma, 'imag')
        
        print(f"H two equal scale factor r and i: {scale_factor_r, scale_factor_i}")
        print(f"CASE 2 sign {sign}")

        # Handle zero displacement cases
        real_k_zero = np.all(W_kappa.real == 0)
        imag_k_zero = np.all(W_kappa.imag == 0)

        # Compute components
        H_rrr = K_box_r @ W_kappa.real / (2*scale_factor_r**2) if not real_k_zero else 0
        H_iir = K_box_i @ W_kappa.real / (2*scale_factor_i**2) if not real_k_zero else 0
        H_rri = K_box_r @ W_kappa.imag / (2*scale_factor_r**2) if not imag_k_zero else 0
        H_iii = K_box_i @ W_kappa.imag / (2*scale_factor_i**2) if not imag_k_zero else 0
        H_ri_r = (mixed_mat @ W_kappa.real)/(scale_factor_r*scale_factor_i*sign) if not real_k_zero else 0
        H_ri_i = (mixed_mat @ W_kappa.imag)/(scale_factor_r*scale_factor_i*sign) if not imag_k_zero else 0

        real_part = H_rrr - H_ri_i - H_iir
        imag_part = H_rri + H_ri_r - H_iii
        return real_part + 1j * imag_part

    def _case_all_distinct(self, sigma, nu, kappa):
        """Case 3: All exponents distinct"""
        print(f"    [CASE3] σ={sigma}, ν={nu}, κ={kappa}")
        W_kappa = self.W_u_dict[kappa]
        W_nu = self.W_u_dict[nu] 
        W_sigma = self.W_u_dict[sigma]

        real_k_zero = np.all(W_kappa.real == 0)
        imag_k_zero = np.all(W_kappa.imag == 0)
        if real_k_zero and imag_k_zero:
            return 0
            
        # Use Option 3 for mixed pairs
        if np.all(W_nu.real+ W_sigma.real == 0):
            K_mixed_pair_rr = self.K
            sign_mixed_r_sigma_r_nu=1
        else:
            K_mixed_pair_rr, scale_factor_r_sigma, scale_factor_r_nu, sign_mixed_r_sigma_r_nu= self._get_K_mixed_pair(sigma, 'real', nu, 'real')

        if np.all(W_nu.real+ W_sigma.imag == 0):
            K_mixed_pair_ir = self.K
            sign_mixed_i_sigma_r_nu= 1

        else:
            K_mixed_pair_ir, scale_factor_i_sigma, scale_factor_r_nu, sign_mixed_i_sigma_r_nu = self._get_K_mixed_pair(sigma, 'imag', nu, 'real')

        if np.all(W_nu.imag+ W_sigma.real == 0):
            K_mixed_pair_ri = self.K
            sign_mixed_r_sigma_i_nu=1
        else:
            K_mixed_pair_ri, scale_factor_r_sigma, scale_factor_i_nu, sign_mixed_r_sigma_i_nu = self._get_K_mixed_pair(sigma, 'real', nu, 'imag')

        if np.all(W_nu.imag+ W_sigma.imag == 0):
            K_mixed_pair_ii = self.K
            sign_mixed_i_sigma_i_nu=1

        else:
            K_mixed_pair_ii, scale_factor_i_sigma, scale_factor_i_nu, sign_mixed_i_sigma_i_nu = self._get_K_mixed_pair(sigma, 'imag', nu, 'imag')


        K_real_sigma, scale_factor_r_sigma = self._get_K_tg(sigma, 'real', 'pos')
        K_imag_sigma, scale_factor_i_sigma= self._get_K_tg(sigma, 'imag', 'pos')
        K_real_nu_pos, scale_factor_r_nu = self._get_K_tg(nu, 'real', 'pos')
        K_imag_nu_pos, scale_factor_i_nu = self._get_K_tg(nu, 'imag', 'pos')
        K_real_nu_neg, scale_factor_r_nu = self._get_K_tg(nu, 'real', 'neg')
        K_imag_nu_neg, scale_factor_i_nu = self._get_K_tg(nu, 'imag', 'neg')

        aux3 = - K_real_sigma - (K_real_nu_pos*(sign_mixed_r_sigma_r_nu+1))/2 - (K_real_nu_neg*(1-sign_mixed_r_sigma_r_nu))/2
        aux4 = K_mixed_pair_rr + self.K 
        K_mixed_pair_rr_real_sigma_real_nu = aux4 +aux3

        aux3 = - K_real_sigma - (K_imag_nu_pos*(sign_mixed_r_sigma_i_nu+1))/2 - (K_imag_nu_neg*(1-sign_mixed_r_sigma_i_nu))/2
        aux4 = K_mixed_pair_ri + self.K 
        K_mixed_pair_ri_real_sigma_imag_nu = aux4 +aux3

        aux3 = - K_imag_sigma - (K_real_nu_pos*(sign_mixed_i_sigma_r_nu+1))/2 - (K_real_nu_neg*(1-sign_mixed_i_sigma_r_nu))/2
        aux4 = K_mixed_pair_ir + self.K  
        K_mixed_pair_ir_imag_sigma_real_nu = aux4 +aux3

        aux3 = - K_imag_sigma - (K_imag_nu_pos*(sign_mixed_i_sigma_i_nu+1))/2 - (K_imag_nu_neg*(1-sign_mixed_i_sigma_i_nu))/2
        aux4 = K_mixed_pair_ii + self.K
        K_mixed_pair_ii_imag_sigma_imag_nu = aux4 +aux3

        print(f"H all different scale factor r_sigma, i_sigma and r_nu, i_nu: {scale_factor_r_sigma, scale_factor_i_sigma, scale_factor_r_nu, scale_factor_i_nu}")

        # Compute all 8 components
        H_rrr = (K_mixed_pair_rr_real_sigma_real_nu @ W_kappa.real)/(scale_factor_r_sigma*scale_factor_r_nu*sign_mixed_r_sigma_r_nu) if not real_k_zero else 0
        H_rii = (K_mixed_pair_ri_real_sigma_imag_nu @ W_kappa.imag)/(scale_factor_r_sigma*scale_factor_i_nu*sign_mixed_r_sigma_i_nu) if not imag_k_zero else 0
        H_iri = (K_mixed_pair_ir_imag_sigma_real_nu @ W_kappa.imag)/(scale_factor_i_sigma*scale_factor_r_nu*sign_mixed_i_sigma_r_nu) if not imag_k_zero else 0
        H_iir = (K_mixed_pair_ii_imag_sigma_imag_nu @ W_kappa.real)/(scale_factor_i_sigma*scale_factor_i_nu*sign_mixed_i_sigma_i_nu) if not real_k_zero else 0
        H_rri = (K_mixed_pair_rr_real_sigma_real_nu @ W_kappa.imag)/(scale_factor_r_sigma*scale_factor_r_nu*sign_mixed_r_sigma_r_nu)  if not imag_k_zero else 0
        H_rir = (K_mixed_pair_ri_real_sigma_imag_nu @ W_kappa.real)/(scale_factor_r_sigma*scale_factor_i_nu*sign_mixed_r_sigma_i_nu) if not real_k_zero else 0
        H_irr = (K_mixed_pair_ir_imag_sigma_real_nu @ W_kappa.real)/(scale_factor_i_sigma*scale_factor_r_nu*sign_mixed_i_sigma_r_nu) if not real_k_zero else 0
        H_iii = (K_mixed_pair_ii_imag_sigma_imag_nu @ W_kappa.imag)/(scale_factor_i_sigma*scale_factor_i_nu*sign_mixed_i_sigma_i_nu) if not imag_k_zero else 0

        real_part = H_rrr - H_rii - H_iri - H_iir
        imag_part = H_rri + H_rir + H_irr - H_iii
        return real_part + 1j * imag_part

    def _get_K_boxplus(self, exponent, part):
        """Compute K_boxplus with caching"""
        key = (exponent, part)
        W_sigma = self.W_u_dict[exponent]
        disp_vector = W_sigma.real if part == 'real' else W_sigma.imag
        scale_factor=(0.24*self.minimum_distance/np.max(np.abs(disp_vector)) if np.any(disp_vector != 0) else 1)

        if key in self.boxplus_cache:

            return self.boxplus_cache[key], scale_factor
        

        K_pos, scale_factor = self._get_K_tg(exponent, part, 'pos')
        K_neg, scale_factor = self._get_K_tg(exponent, part, 'neg')
            
        K_boxplus = K_pos + K_neg - 2 * self.K
        
        self.boxplus_cache[key] = K_boxplus
        return K_boxplus, scale_factor

    def _get_K_tg(self, exponent, part, sign):
        """Compute tangent stiffness matrix with caching (Option 1)"""
        key = (exponent, part, sign)
        sign_mixed=1
        W_sigma = self.W_u_dict[exponent]
        disp_vector = W_sigma.real if part == 'real' else W_sigma.imag
        if np.any(disp_vector != 0):
            scale_factor=(0.24*self.minimum_distance)/np.max(np.abs(disp_vector))
        else:
            scale_factor=1
        #print(scale_factor)
        print(f"SIGN CASE 1 {sign_mixed}")

        if key in self.tangent_cache:
            return self.tangent_cache[key], scale_factor 

        
        # If displacement is zero, return linear stiffness matrix K
        if np.all(disp_vector == 0):
            self.tangent_cache[key] = self.K
            print("passei por aqui")
            return self.K, scale_factor

        else:
            # Otherwise compute tangent stiffness using Option 1
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
            #print(scale_factor)
            return K_tg, scale_factor

    def _get_K_mixed(self, exponent):
        """Compute mixed term matrix for case 2 with caching (Option 2)"""
        key = ('mixed', exponent)
        W = self.W_u_dict[exponent]
        disp_real = W.real
        disp_imag = W.imag
        scale_factor_r=(0.24*self.minimum_distance/np.max(np.abs(disp_real)) if np.any(disp_real != 0) else 1)
        scale_factor_i=(0.24*self.minimum_distance/np.max(np.abs(disp_imag)) if np.any(disp_imag != 0) else 1)
        if key in self.mixed_cache:

            return self.mixed_cache[key], scale_factor_r, scale_factor_i, self.sign_mixed_cache[key]

        if np.all(disp_real+disp_imag == 0):
            K_mixed = self.K
            sign=1

        else:
            K_mixed, scale_factor_r, scale_factor_i, sign = self._get_K_tg_for_disp(W, exponent)

        K_real, scale_factor_r = self._get_K_tg(exponent, 'real', 'pos')
        K_imag_pos, scale_factor_i = self._get_K_tg(exponent, 'imag', 'pos')
        K_imag_neg, scale_factor_i = self._get_K_tg(exponent, 'imag', 'neg')
        print(f"scale_factor_r: {scale_factor_r}")
        print(f"scale_factor_i: {scale_factor_i}")
        aux1=self.K - ((sign+1)*K_imag_pos)/2 - ((1-sign)*K_imag_neg)/2
        aux2=K_mixed - K_real
        mixed_mat = aux1+aux2

        self.mixed_cache[key] = mixed_mat
        self.sign_mixed_cache[key]=sign
        return mixed_mat, scale_factor_r, scale_factor_i, sign

    def _get_K_mixed_pair(self, exp1, part1, exp2, part2):
        """Compute mixed term matrix for case 3 with caching (Option 3)"""
        key = (exp1, part1, exp2, part2)
        W_sigma = self.W_u_dict[exp1]
        W_nu = self.W_u_dict[exp2]
        disp_sigma = W_sigma.real if part1 == 'real' else W_sigma.imag
        disp_nu = W_nu.real if part2 == 'real' else W_nu.imag
        scale_factor_sigma=(0.24*self.minimum_distance/np.max(np.abs(disp_sigma)) if np.any(disp_sigma != 0) else 1)
        scale_factor_nu=(0.24*self.minimum_distance/np.max(np.abs(disp_nu)) if np.any(disp_nu != 0) else 1)
        sign_mixed=1
        disp_combined_plus = scale_factor_sigma*disp_sigma + scale_factor_nu*disp_nu



        if max(abs(disp_combined_plus)) >= 0.24*self.minimum_distance:
            sign_mixed=1

        else:
            #compute difference of displacements
            sign_mixed=-1

        if key in self.mixed_pair_cache:
            print(f"    [K_mixed_pair] CACHE HIT for {key}")
            return self.mixed_pair_cache[key], scale_factor_sigma, scale_factor_nu, self.sign_mixed_pair_cache[key]
        

        else:
            # Use Option 3 for mixed pairs
            handler = TangentStiffnessMatrixHandler(
                option=3,
                base_input_filename=self.base_input_filename,
                non_embedded_nodes=self.non_embedded_nodes,
                exponent_table=self.exponent_generator,
                exponent1=exp1,
                exponent2=exp2,
                minimum_distance= self.minimum_distance,
                part1=part1,
                part2=part2,
                W_u_dict=self.W_u_dict
            )
            K_mixed_pair= handler.compute_or_load()
            print(f"//////////////////////////////sign_mixed AQUI 3 CASE: {sign_mixed}///////////////////////////////////")
            
            self.mixed_pair_cache[key] = K_mixed_pair
            self.sign_mixed_pair_cache[key] = sign_mixed
            return K_mixed_pair, scale_factor_sigma, scale_factor_nu, sign_mixed

    def _get_K_tg_for_disp(self, disp_vector, exponent):
        """Compute tangent stiffness for arbitrary displacement (Option 2)"""
        W = self.W_u_dict[exponent]
        disp_real = W.real
        disp_imag = W.imag  
        scale_factor_r=(0.24*self.minimum_distance/np.max(np.abs(disp_real)) if np.any(disp_real != 0) else 1)
        scale_factor_i=(0.24*self.minimum_distance/np.max(np.abs(disp_imag)) if np.any(disp_imag != 0) else 1)
        sign=1
        disp_combined_plus = disp_real*scale_factor_r +  disp_imag*scale_factor_i


        if max(abs(disp_combined_plus)) >= 0.24*self.minimum_distance:
            sign=1

        else:
            sign=-1

        if np.all(disp_vector.imag == 0):
            K_tg, scale_factor_r = self._get_K_tg(exponent, 'real', 'pos')
            scale_factor_i=1
            sign=1
            return K_tg, scale_factor_r, scale_factor_i, sign
        if np.all(disp_vector.real == 0):
            K_tg, scale_factor_i = self._get_K_tg(exponent, 'imag', 'pos')
            scale_factor_r=1
            sign=1
            return K_tg, scale_factor_r, scale_factor_i, sign 
        else:
            # Use Option 2 for arbitrary displacement
            handler = TangentStiffnessMatrixHandler(
                option=2,
                base_input_filename=self.base_input_filename,
                non_embedded_nodes=self.non_embedded_nodes,
                exponent_table=self.exponent_generator,
                exponent_real=exponent,
                exponent_imag=exponent,
                displacement=disp_vector.flatten().tolist(), 
                minimum_distance= self.minimum_distance
            )
            K_tg= handler.compute_or_load()
            print(f"//////////////////////////////sign_mixed AQUI 2 CASE: {sign}///////////////////////////////////")
            return K_tg, scale_factor_r, scale_factor_i, sign

    def compute_triple_contribution(self, sigma, nu, kappa):
        """
        Compute the contribution for a specific triple (sigma, nu, kappa).
        Routes to the appropriate case method based on exponent equality.
        """
        if sigma == nu == kappa:
            return self._case_all_equal(sigma)
        elif sigma == nu:
            return self._case_two_equal(sigma, kappa)
        elif sigma == kappa:
            return self._case_two_equal(sigma, nu)
        elif nu == kappa:
            return self._case_two_equal(nu, sigma)
        else:
            return self._case_all_distinct(sigma, nu, kappa)



class CubicCombinationFinder:
    def __init__(self, exponent_table, order_table, exponent_dict):
        self.exponent_table = exponent_table
        self.order_table = order_table
        self.exponent_dict = exponent_dict

    def case1(self, gamma):
        """
        Solves γ = 3α for monomial exponent vector α
        Returns solution where σ = ν = κ = α if valid
        """

        gamma_arr = np.array(gamma)
        if sum(gamma) % 3 != 0: return []
        a = gamma_arr // 3
        if not np.array_equal(3*a, gamma_arr): return []
        a_tuple = tuple(a)
        if a_tuple not in self.exponent_dict: return []
        return [{'sigma': a_tuple, 'nu': a_tuple, 'kappa': a_tuple}]

    def case2_and_3(self, gamma):
        """
        Solves γ = 2α + β for distinct monomials α, β
        Returns solutions of form σ = α, ν = α, κ = β
        """

        gamma_arr = np.array(gamma)
        gamma_order = sum(gamma)
        candidate_df = self.exponent_table[self.exponent_table['order'] < gamma_order]
        combinations = []
        for a_tuple in candidate_df['exponents']:
            a_arr = np.array(a_tuple)
            b_arr = gamma_arr - 2*a_arr
            if np.any(b_arr < 0): continue
            if (b_tuple := tuple(b_arr)) not in self.exponent_dict: continue
            if a_tuple == b_tuple: continue
            combinations.append({'sigma': a_tuple, 'nu': a_tuple, 'kappa': b_tuple})
        return combinations

    def case4(self, gamma):
        gamma_arr, gamma_order = np.array(gamma), sum(gamma)
        candidate_df = self.exponent_table[self.exponent_table['order'] < gamma_order]
        if candidate_df.empty: return []
        candidate_pairs = list(zip(candidate_df['exponents'], candidate_df['global_index']))
        candidate_pairs_sorted = sorted(candidate_pairs, key=lambda x: x[1])
        combinations = []
        for i, (a, a_idx) in enumerate(candidate_pairs_sorted):
            rem1 = gamma_arr - np.array(a)
            if np.any(rem1 < 0): continue
            self._case4_inner_loop(i, candidate_pairs_sorted, rem1, combinations, a, a_idx)
        return combinations

    def _case4_inner_loop(self, i, pairs, rem1, combs, a, a_idx):
        """
        Helper for case4: Completes γ = α + β + γ decomposition
        Finds valid triplets with strictly increasing global indices
        """
        n = len(pairs)
        for j in range(i+1, n):
            b, b_idx = pairs[j]
            if np.any((rem2 := rem1 - np.array(b)) < 0): continue
            if (c_tuple := tuple(rem2)) not in self.exponent_dict: continue
            c_idx = self.exponent_dict[c_tuple] + 1
            if a == b or a == c_tuple or b == c_tuple: continue
            if not (a_idx < b_idx < c_idx): continue
            srt = sorted([(a, a_idx), (b, b_idx), (c_tuple, c_idx)], key=lambda x: x[1])
            combs.append({'sigma': srt[0][0], 'nu': srt[1][0], 'kappa': srt[2][0]})



def build_symbolic_polynomial(compute_H_terms, symbolic_vars=('z1', 'z2', 'z3', 'z4')):
    """
    Builds a symbolic polynomial from computed H coefficients for all combinations of basis exponents.
    
    Args:
        compute_H_terms: Instance of ComputeHterms with computed coefficients
        symbolic_vars: Tuple of symbolic variable names (default: ('z1','z2','z3','z4'))
    
    Returns:
        sympy.Expr: Symbolic polynomial representing the sum of all H terms
    """
    from sympy import symbols, Mul, Add, Integer
    from sympy.core.expr import Expr
    import numpy as np

    # Define symbolic variables
    z1, z2, z3, z4 = symbols(' '.join(symbolic_vars))
    z = [z1, z2, z3, z4]
    
    polynomial = 0  # Initialize the polynomial
    
    # Get basis exponents from W_u_dict keys
    basis_exponents = list(compute_H_terms.W_u_dict.keys())
    n = len(basis_exponents)
    print(f"Building symbolic polynomial from {n} basis exponents -> {n**3} triples")
    
    for i, sigma in enumerate(basis_exponents):
        for j, nu in enumerate(basis_exponents):
            for k, kappa in enumerate(basis_exponents):
                # Skip if any exponent is not in W_u_dict (shouldn't happen, but safe)
                if (sigma not in compute_H_terms.W_u_dict or 
                    nu not in compute_H_terms.W_u_dict or 
                    kappa not in compute_H_terms.W_u_dict):
                    continue
                
                # Compute contribution for this triple
                try:
                    contribution = compute_H_terms.compute_triple_contribution(sigma, nu, kappa)
                except KeyError as e:
                    print(f"Skipping triple ({sigma}, {nu}, {kappa}): {str(e)}")
                    continue
                
                # Skip if contribution is zero (or negligible)
                if np.allclose(contribution, 0):
                    continue
                
                # Build monomials for each exponent
                def build_monomial(exponent):
                    """Constructs z1^a * z2^b * z3^c * z4^d from exponent tuple"""
                    monomial = 1
                    for idx, exp in enumerate(exponent):
                        if exp > 0:
                            monomial *= z[idx]**exp
                    return monomial
                
                term_sigma = build_monomial(sigma)
                term_nu = build_monomial(nu)
                term_kappa = build_monomial(kappa)
                
                # Construct full term: H_{σ,ν,κ} * (z^σ * z^ν * z^κ)
                full_term = contribution * term_sigma * term_nu * term_kappa
                polynomial += full_term
                
    return polynomial


