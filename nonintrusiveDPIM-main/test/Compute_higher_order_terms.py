import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from StEP_validation_with_complex_numbers_abaqus_G_terms import ComputeGterms 
from StEP_validation_complex_numbers_abaqus_H_terms import ComputeHterms

class HigherOrderSolver:
    """Solves higher-order terms in the dynamic power iteration method (DPIM) expansion."""

    def __init__(self, dpim_instance):
        """
        Initialize the higher-order solver with a DPIM instance.
        
        Args:
            dpim_instance: Instance of the DynamicPowerIterationMethod class
        """
        self.dpim = dpim_instance
        self.current_gamma = None
        self.current_i_gl = None
        self.current_s = None
        self.is_resonant = False
        self.resonant_modes = None
        
    def process_order(self, order, n_modes):
        """
        Process all terms of a given order in the expansion.
        
        Args:
            order (int): The order to process (e.g., 2 for second-order terms)
            n_modes (int): Number of modes in the system
        """
        order_info = self.dpim.mexp.order_table
        order_info = order_info[order_info['order'] == order]
        if order_info.empty: 
            print(f"WARNING: No terms for order {order}, skipping")
            return

        first_idx = order_info.iloc[0]['first_index']
        #print(f"first_idx: {first_idx}")
        last_idx = order_info.iloc[0]['last_index']
        #print(f"last_idx: {last_idx}")
        for i_gl in range(first_idx, last_idx + 1):
            #print(f"i_gl: {i_gl}")
            self._process_term(i_gl, n_modes)

    def _process_term(self, i_gl, n_modes):
        """
        Process a single term in the expansion.
        
        Args:
            i_gl (int): Global index of the term to process
            n_modes (int): Number of modes in the system
        """
        self.current_i_gl = i_gl
        #print(f"self.current_i_gl: {self.current_i_gl}")
        row = self.dpim.mexp.exponent_table.iloc[i_gl-1]
        #print(f"row: {row}")
        if self._should_use_conjugate(row):
            print("ENTREI AQUI CONJUGATE")
            self._apply_conjugate_symmetry(row, n_modes)
        else:
            print("CALCULAR SEM CONJUGATE")
            self._process_single_term()

    def _should_use_conjugate(self, row):
        """
        Determine if conjugate symmetry should be used for this term.
        
        Args:
            row: Row from exponent table for current term
            
        Returns:
            bool: True if conjugate solution should be used
        """
        #print(f"row['global_index_cc']: {row['global_index_cc']} and row['global_index'] {row['global_index']}")
        return row['global_index_cc'] < row['global_index']

    def _apply_conjugate_symmetry(self, row, n_modes):
        """
        Apply conjugate symmetry to compute solution from its conjugate pair.
        
        Args:
            row: Row from exponent table for current term
            n_modes (int): Number of modes in the system
        """
        cc_idx = int(row['global_index_cc'])
        cc_row = self.dpim.mexp.exponent_table[
            self.dpim.mexp.exponent_table['global_index'] == cc_idx
        ]
        #print(f"cc_row : {cc_row}")
        if cc_row.empty: 
            self._fallback_to_normal(self.current_i_gl)
            return

        cc_gamma = tuple(cc_row.iloc[0]['exponents'])
        if cc_gamma not in self.dpim.W_u_dict:
            self._fallback_to_normal(self.current_i_gl)
            return

        self._set_conjugate_solution(row, cc_gamma, cc_idx, n_modes)

    def _fallback_to_normal(self, i_gl):
        """
        Fallback to normal computation when conjugate solution is unavailable.
        
        Args:
            i_gl (int): Global index of term to compute normally
        """
        print(f"WARNING: Missing conjugate solutions. Computing term {i_gl} normally.")
        self._process_single_term(i_gl)

    def _set_conjugate_solution(self, row, cc_gamma, cc_idx, n_modes):
        """
        Set solution using conjugate symmetry from existing solution.
        
        Args:
            row: Row from exponent table for current term
            cc_gamma: Exponent vector of conjugate term
            cc_idx: Global index of conjugate term
            n_modes (int): Number of modes in the system
        """
        col_idx = cc_idx - 1
        u_cc = self.dpim.Wu[:, col_idx]
        v_cc = self.dpim.Wv[:, col_idx]
        r_cc = self.dpim.f[:, col_idx]
        
        half = n_modes // 2
        r_swapped = np.concatenate([r_cc[half:], r_cc[:half]])
        
        current_col = row['global_index'] - 1
        self.dpim.Wu[:, current_col] = np.conj(u_cc)
        self.dpim.Wv[:, current_col] = np.conj(v_cc)
        self.dpim.f[:, current_col] = np.conj(r_swapped)
        self.dpim.W_u_dict[tuple(row['exponents'])] = self.dpim.Wu[:, current_col]

    def _process_single_term(self):
        """
        Compute solution for a single term without using conjugate symmetry.

        """
        self._setup_current_term()
        self._detect_resonance()
        self._compute_nonlinear_terms()
        mu_gamma = self.dpim._compute_mu_gamma(self.current_gamma)
        self._solve_homological_equation(mu_gamma)
        self._compute_velocity_mapping(mu_gamma)

    def _setup_current_term(self):
        """Initialize current term parameters from exponent table."""
        self.current_gamma = np.array(
            self.dpim.mexp.exponent_table.iloc[self.current_i_gl-1]['exponents']
        )
        #print(f"current term: {self.current_gamma}")
        self.current_s = np.dot(self.dpim.lambda_tilde, self.current_gamma)
        #print(f"self.current_s: {self.current_s}")

    def _detect_resonance(self):
        """Detect if current term is resonant with any system modes."""
        self.resonant_modes = self.dpim._detect_resonances(
            self.current_gamma, self.current_s
        )
        
        self.is_resonant = self.resonant_modes is not None
        #print(f"self.is_resonant: {self.is_resonant}")

    def _compute_nonlinear_terms(self):
        """
        Compute nonlinear terms G and H for the current exponent vector.
        
        The RHS of the homological equation is computed as:
            rhs = ξ(γ) - ΣG(γ) - ΣH(γ)
        """
        g_computer = ComputeGterms(
            self.dpim.K, self.dpim.W_u_dict,
            self.dpim.non_encased_nodes,
            self.dpim.base_input_filename,
            self.dpim.mexp, self.dpim.n_modes,
            self.dpim.num_forcing, self.dpim.minimum_distance
        )
        sum_G = g_computer.compute_G(self.current_gamma)
        
        h_computer = ComputeHterms(
            self.dpim.K, self.dpim.W_u_dict,
            self.dpim.non_encased_nodes,
            self.dpim.base_input_filename,
            self.dpim.mexp, self.dpim.n_modes,
            self.dpim.num_forcing, self.dpim.minimum_distance
        )
        sum_H = h_computer.compute_H(self.current_gamma)
        xi=self.dpim._compute_xi_gamma(
            self.current_gamma, self.current_i_gl
        )
        #print(f"xi : {xi}")

        with open('rhs_equation.txt','a') as file:
            file.write(f"\ngamma=\n{self.current_gamma}")
            file.write(f"\nG+H=\n {sum_G+sum_H}")
            file.write(f"\nxi=\n {xi}")

        self.rhs = xi  -sum_H - sum_G 
        print(f"///////////////////////////RHS {self.rhs}//////////////////////////////////////")

    def _solve_homological_equation(self, mu_gamma):
        """
        Solve the homological equation for the current term.
        
        Args:
            mu_gamma: Precomputed μ(γ) value for current exponent vector
        """
        self.dpim._solve_homological_equation(
            self.current_gamma, self.current_s, self.rhs,
            self.current_i_gl, mu_gamma,
            self.is_resonant, self.resonant_modes
        )

    def _compute_velocity_mapping(self, mu_gamma):
        """
        Compute and store the velocity mapping for the current term.
        
        Args:
            mu_gamma: Precomputed μ(γ) value for current exponent vector
        """
        self.dpim._compute_velocity_mapping(
            self.current_gamma, self.current_i_gl, mu_gamma
        )