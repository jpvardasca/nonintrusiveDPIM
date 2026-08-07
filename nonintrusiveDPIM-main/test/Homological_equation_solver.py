import numpy as np

class HomologicalEquationSolver_normal_style:
    """Solves homological equations for normal style parameterization (handles both resonant and non-resonant cases)."""
    
    def __init__(self, dpim_instance):
        """Initialize the solver with a DPIM (Dynamic Parameterization of Invariant Manifolds) instance.
        
        Args:
            dpim_instance: An instance of the DPIM class containing system matrices and modes
        """
        self.dpim = dpim_instance
        
    def solve(self, gamma, s, rhs, i_gl, is_resonant=False, resonant_modes=None, mu_gamma=None):
        """Main solver method that routes to either resonant or non-resonant solver.
        
        Args:
            gamma: Multi-index vector
            s: Complex frequency parameter
            rhs: Right-hand side vector of the homological equation
            i_gl: Index for storing results
            is_resonant: Flag indicating if this is a resonant case
            resonant_modes: List of resonant mode indices (for resonant cases)
            mu_gamma: Correction term for resonant cases
            
        Returns:
            Tuple of (Wu_gamma, f_gamma) solutions
        """
        print(f"\n{'='*80}")
        print(f"SOLVING HOMOLOGICAL EQUATION FOR gamma={gamma}, i_gl={i_gl}, s={s:.4f}")
        print(f"Resonant: {is_resonant}, Resonant modes: {resonant_modes}")
        print(f"mu_gamma type: {type(mu_gamma)}, norm: {np.linalg.norm(mu_gamma) if mu_gamma is not None else 'None'}")
        print(f"RHS norm: {np.linalg.norm(rhs)}")
        
        if is_resonant:
            print("---------------- RESONANT CASE ----------------------------")
            return self._solve_resonant_case_matrix(
                gamma, s, rhs, i_gl, resonant_modes, mu_gamma
            )
        else:
            print("---------------- NONRESONANT CASE -------------------------")
            return self._solve_non_resonant_case(
                gamma, s, rhs, i_gl
            )
    
    def _solve_resonant_case_matrix(self, gamma, s, rhs, i_gl, resonant_modes, mu_gamma):
        """Solves the homological equation for resonant cases using a block matrix approach.
        
        Args:
            gamma: Multi-index vector
            s: Complex frequency parameter
            rhs: Right-hand side vector
            i_gl: Index for storing results
            resonant_modes: List of resonant mode indices
            mu_gamma: Correction term for resonant cases
            
        Returns:
            Tuple of (Wu_gamma, f_gamma) solutions
        """
        # Get resonant mode information
        Phi_R, Psi_R, lambda_R = self._get_resonant_matrices(resonant_modes)
        n_dof = self.dpim.n_dof
        n_R = len(resonant_modes)
        
        # Build the full block matrix system
        big_mat = self._build_resonant_block_matrix(s, Phi_R, Psi_R, lambda_R)
        
        # Build the right-hand side vector with mu_gamma correction
        big_rhs = np.zeros(n_dof + n_R, dtype=np.complex128)
        big_rhs[:n_dof] = rhs  # Top block: Ξ_γ
        
        # Bottom block: -X_{ℛ}^{V*} M μ_γ
        Psi_star = Psi_R.conj().T
        M_mu = self.dpim.M @ mu_gamma
        
        big_rhs[n_dof:] = -Psi_star @ M_mu
        
        with open('Homological_equation.txt','a') as file:
            file.write(f"\ngamma=\n {gamma}")
            file.write(f"\nbig_RHS=\n {big_rhs}")
            file.write(f"\nbig_mat=\n {big_mat}")

        # Solve the full system
        print("\nSOLVING SYSTEM...")
        try:
            solution = np.linalg.solve(big_mat, big_rhs)
            print("System solved successfully!")
        except np.linalg.LinAlgError as e:
            print(f"SOLVE ERROR: {str(e)}")
            # Fallback to least-squares solution
            solution = np.linalg.lstsq(big_mat, big_rhs, rcond=None)[0]
            
        Wu_gamma = solution[:n_dof]
        f_R = solution[n_dof:]
        # Non-resonant coefficients are explicitly zero
        f_gamma = np.zeros(self.dpim.n_modes, dtype=np.complex128)
        f_gamma[resonant_modes] = f_R
        
        self._store_results(gamma, i_gl, Wu_gamma, f_gamma)
        return Wu_gamma, f_gamma

    def _build_resonant_block_matrix(self, s, Phi_R, Psi_R, lambda_R):
        """Constructs the block matrix for resonant case solving.
        
        Args:
            s: Complex frequency parameter
            Phi_R: Right eigenvectors for resonant modes
            Psi_R: Left eigenvectors for resonant modes
            lambda_R: Diagonal matrix of eigenvalues for resonant modes
            
        Returns:
            The constructed block matrix
        """
        n_dof = self.dpim.n_dof
        n_R = Phi_R.shape[1]
        total_size = n_dof + n_R
        big_mat = np.zeros((total_size, total_size), dtype=np.complex128)
        
        # Top-left: s^2M + sC + K
        L_combined = self.dpim._get_linear_operator(s)
        big_mat[:n_dof, :n_dof] = L_combined[:, :self.dpim.n_dof]
        
        # Top-right: MΦ_ℛ(sI + Λ_ℛ) + CΦ_ℛ
        s_plus_lambda = s*np.eye(n_R) + lambda_R
        top_right_block = (
            self.dpim.M @ (Phi_R * s_plus_lambda) 
            + self.dpim.C @ Phi_R
        )
        big_mat[:n_dof, n_dof:] = top_right_block
        
        # Bottom-left: (sI + Λ_ℛ)X_ℛ^{V*}M + X_ℛ^{V*}C
        Psi_star = Psi_R.conj().T
        Psi_star_M = Psi_star @ self.dpim.M
        bottom_left_block = (
            s_plus_lambda @ Psi_star_M 
            + Psi_star @ self.dpim.C
        )
        big_mat[n_dof:, :n_dof] = bottom_left_block
        
        # Bottom-right: X_ℛ^{V*}MΦ_ℛ
        bottom_right_block = Psi_star @ self.dpim.M @ Phi_R
        big_mat[n_dof:, n_dof:] = bottom_right_block
        
        return big_mat

    def _get_resonant_matrices(self, resonant_modes):
        """Extracts matrices for resonant modes.
        
        Args:
            resonant_modes: List of resonant mode indices
            
        Returns:
            Tuple of (Phi_R, Psi_R, lambda_R) matrices for resonant modes
        """
        Phi_R = self.dpim.Phi[:, resonant_modes]
        Psi_R = self.dpim.Psi[:, resonant_modes]
        lambda_R = self.dpim.lambda_[resonant_modes]
        lambda_R=  np.diag(lambda_R)
        return Phi_R, Psi_R, lambda_R

    def _solve_non_resonant_case(self, gamma, s, rhs, i_gl):
        """Solves the homological equation for non-resonant cases.
        
        Args:
            gamma: Multi-index vector
            s: Complex frequency parameter
            rhs: Right-hand side vector
            i_gl: Index for storing results
            
        Returns:
            Tuple of (Wu_gamma, f_gamma) solutions
        """
        L_combined = self.dpim._get_linear_operator(s)

        Wu_gamma = np.linalg.solve(
            L_combined[:, :self.dpim.n_dof], rhs
        )
        
        with open('Homological_equation.txt','a') as file:
            file.write(f"\ngamma=\n {gamma}")
            file.write(f"\nL_combined=\n {L_combined}")
            file.write(f"\nrhs=\n {rhs}")

        f_gamma = np.zeros(self.dpim.n_modes, dtype=np.complex128)
        self._store_results(gamma, i_gl, Wu_gamma, f_gamma)
        return Wu_gamma, f_gamma

    def _store_results(self, gamma, i_gl, Wu_gamma, f_gamma):
        """Stores the solution results in the DPIM instance.
        
        Args:
            gamma: Multi-index vector
            i_gl: Index for storing results
            Wu_gamma: Nonlinear mapping solution
            f_gamma: Reduced dynamics coefficients
        """
        print(f"Storing results for gamma={gamma}, i_gl={i_gl}")
        print(f"Wu_gamma norm: {np.linalg.norm(Wu_gamma)}")
        print(f"f_gamma: {f_gamma} (norm: {np.linalg.norm(f_gamma)})")
        
        self.dpim.Wu[:, i_gl-1] = Wu_gamma
        gamma_tuple = tuple(gamma)
        self.dpim.W_u_dict[gamma_tuple] = self.dpim.Wu[:, i_gl-1]
        self.dpim.f[:, i_gl-1] = f_gamma
        print(f"Stored in Wu[:, {i_gl-1}] and f[:, {i_gl-1}]")






class HomologicalEquationSolver_graph_style:
    """Solves homological equations for graph style parameterization (always treats as resonant case)."""
    
    def __init__(self, dpim_instance):
        """Initialize the solver with a DPIM instance.
        
        Args:
            dpim_instance: An instance of the DPIM class containing system matrices and modes
        """
        self.dpim = dpim_instance
        
    def solve(self, gamma, s, rhs, i_gl, n_modes, mu_gamma):
        """Main solver method for graph style parameterization.
        
        Args:
            gamma: Multi-index vector
            s: Complex frequency parameter
            rhs: Right-hand side vector
            i_gl: Index for storing results
            n_modes: Number of modes to consider
            mu_gamma: Correction term
            
        Returns:
            Tuple of (Wu_gamma, f_gamma) solutions
        """
        print("---------------- HOMOLOGICAL EQUATION WITH GRAPH STYLE ----------------------------")
        resonant_modes = np.arange(n_modes)  
        print(f"resonante modes: {resonant_modes}")
        return self._solve_case_matrix(gamma, s, rhs, i_gl, resonant_modes, mu_gamma)

    def _solve_case_matrix(self, gamma, s, rhs, i_gl, resonant_modes, mu_gamma):
        """Solves the homological equation using a block matrix approach (graph style).
        
        Args:
            gamma: Multi-index vector
            s: Complex frequency parameter
            rhs: Right-hand side vector
            i_gl: Index for storing results
            resonant_modes: List of resonant mode indices
            mu_gamma: Correction term
            
        Returns:
            Tuple of (Wu_gamma, f_gamma) solutions
        """
        # Extract resonant modes matrices
        print(f"dpim n modes line 257 homological equation: {len(resonant_modes)}")
        Phi_R = self.dpim.Phi[:, resonant_modes]

        print(f"dpim n modes line 260 homological equation: {Phi_R}")

        Psi_R = self.dpim.Psi[:, resonant_modes]
        lambda_R = self.dpim.lambda_[resonant_modes]
        lambda_R=  np.diag(lambda_R)
        n_R = len(resonant_modes)
        n_dof = self.dpim.n_dof
        
        # Build full coefficient matrix
        big_mat = self._build_resonant_block_matrix(s, Phi_R, Psi_R, lambda_R)
        
        # Build full RHS vector
        big_rhs = np.zeros(n_dof + n_R, dtype=np.complex128)
        big_rhs[:n_dof] = rhs  # Top block: Ξ_γ
        
        # Bottom block: -X_{ℛ}^{V*} M μ_γ
        Psi_star = Psi_R.conj().T
        M_mu = self.dpim.M @ mu_gamma
        
        big_rhs[n_dof:] = -Psi_star @ M_mu
        
        with open('Homological_equation.txt','a') as file:
            file.write(f"\ngamma=\n {gamma}")
            file.write(f"\nbig_RHS=\n {big_rhs}")
            file.write(f"\nbig_mat=\n {big_mat}")

        # Solve the full system
        print("\nSOLVING SYSTEM...")
        try:
            solution = np.linalg.solve(big_mat, big_rhs)
            print("System solved successfully!")
        except np.linalg.LinAlgError as e:
            print(f"SOLVE ERROR: {str(e)}")
            # Fallback to least-squares solution
            solution = np.linalg.lstsq(big_mat, big_rhs, rcond=None)[0]
            
        Wu_gamma = solution[:n_dof]
        f_R = solution[n_dof:]
        # Non-resonant coefficients are explicitly zero
        f_gamma = np.zeros(self.dpim.n_modes, dtype=np.complex128)
        f_gamma[resonant_modes] = f_R
        
        self._store_results(gamma, i_gl, Wu_gamma, f_gamma)
        return Wu_gamma, f_gamma

    def _build_resonant_block_matrix(self, s, Phi_R, Psi_R, lambda_R):
        """Constructs the block matrix for graph style solving.
        
        Args:
            s: Complex frequency parameter
            Phi_R: Right eigenvectors for resonant modes
            Psi_R: Left eigenvectors for resonant modes
            lambda_R: Diagonal matrix of eigenvalues for resonant modes
            
        Returns:
            The constructed block matrix
        """
        n_dof = self.dpim.n_dof
        n_R = Phi_R.shape[1]
        print(f"line 315 homological equation : {n_R}")
        total_size = n_dof + n_R
        big_mat = np.zeros((total_size, total_size), dtype=np.complex128)
        
        # Top-left: s^2M + sC + K
        L_combined = self.dpim._get_linear_operator(s)
        big_mat[:n_dof, :n_dof] = L_combined[:, :self.dpim.n_dof]
        
        # Top-right: MΦ_ℛ(sI + Λ_ℛ) + CΦ_ℛ
        s_plus_lambda = s*np.eye(n_R) + lambda_R
        top_right_block = (
            self.dpim.M @ (Phi_R @ s_plus_lambda) 
            + self.dpim.C @ Phi_R
        )
        big_mat[:n_dof, n_dof:] = top_right_block
        
        # Bottom-left: (sI + Λ_ℛ)X_ℛ^{V*}M + X_ℛ^{V*}C
        Psi_star = Psi_R.conj().T
        Psi_star_M = Psi_star @ self.dpim.M
        bottom_left_block = (
            s_plus_lambda @ Psi_star_M 
            + Psi_star @ self.dpim.C
        )
        big_mat[n_dof:, :n_dof] = bottom_left_block
        
        # Bottom-right: X_ℛ^{V*}MΦ_ℛ
        bottom_right_block = Psi_star @ self.dpim.M @ Phi_R
        big_mat[n_dof:, n_dof:] = bottom_right_block
        
        return big_mat

    def _store_results(self, gamma, i_gl, Wu_gamma, f_gamma):
        """Stores the solution results in the DPIM instance.
        
        Args:
            gamma: Multi-index vector
            i_gl: Index for storing results
            Wu_gamma: Nonlinear mapping solution
            f_gamma: Reduced dynamics coefficients
        """
        print(f"gamma homologica equatoion: {gamma}")
        print(f"nonlinear mapping homologica equatoion: {Wu_gamma}")
        print(f"reduced system homologica equatoion: {f_gamma}")
        self.dpim.Wu[:, i_gl-1] = Wu_gamma
        gamma_tuple = tuple(gamma)
        self.dpim.W_u_dict[gamma_tuple] = Wu_gamma
        self.dpim.f[:, i_gl-1] = f_gamma