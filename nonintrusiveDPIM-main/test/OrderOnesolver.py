
import numpy as np

from Compute_OrderOneTerms import OrderOneTerms


class OrderOneSolver:
    def __init__(self, dpim):
        """Initialize the first-order term solver with DPIM system data.
        
        Args:
            dpim: Data structure containing system matrices and parameters
        """
        self.dpim = dpim

    def _create_order_one_solver(self):
        """Create a dedicated solver instance for first-order terms.
        
        Returns:
            OrderOneTerms: Solver instance initialized with system parameters
        """
        return OrderOneTerms(
            M=self.dpim.M, C=self.dpim.C, K=self.dpim.K, 
            Phi=self.dpim.Phi, Psi=self.dpim.Psi,
            lambda_=self.dpim.lambda_, lambda_tilde=self.dpim.lambda_tilde,
            force_vectors=self.dpim.force_vectors, n_modes=self.dpim.n_modes,
            n_forcing=self.dpim.n_forcing, resonance_tol=self.dpim.resonance_tol
        )

    def _store_order_one_results(self, n_total, Wu_list, Wv_list, f_list):
        """Store computed first-order solutions in the DPIM data structure.
        
        Args:
            n_total: Total number of variables (modes + forcing)
            Wu_list: List of Wu solutions for each first-order term
            Wv_list: List of Wv solutions for each first-order term
            f_list: List of forcing term solutions
        """
        for j, exp in enumerate(np.eye(n_total, dtype=int)):
            row = self.dpim.query.find_combination(tuple(exp))
            if row is not None:
                i_gl = row['global_index'].values[0] - 1
                self.dpim.Wu[:, i_gl] = Wu_list[j]
                self.dpim.Wv[:, i_gl] = Wv_list[j]
                self.dpim.f[:, i_gl] = f_list[j]
                self.dpim.W_u_dict[tuple(exp)] = Wu_list[j]

    def _update_resonance_relations(self, res_relations, res_terms):
        """Update resonance tracking structures with detected resonances.
        
        Args:
            res_relations: Detected resonance relationships
            res_terms: Set of resonant terms to track
        """
        for mode, terms in res_relations.items():
            self.dpim.resonance_relations[mode].extend(terms)
        self.dpim.resonant_terms.update(res_terms)

    def solve_order_one(self):
        """Main driver method for solving first-order terms."""
        print("\n=== Solving order 1 ===")
        solver = self._create_order_one_solver()
        Wu_list, Wv_list, f_list, res_relations, res_terms = solver.solve()
        self._store_order_one_results(self.dpim.n_total, Wu_list, Wv_list, f_list)
        self._update_resonance_relations(res_relations, res_terms)