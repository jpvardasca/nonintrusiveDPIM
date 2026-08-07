
import h5py
import numpy as np
from scipy.sparse import csr_matrix
import time
import scipy.sparse as sp
from scipy.io import mmwrite


from eigenproblem_new import eigensolutions_damped
from Multiindex_exponents import MultiExponentGenerator
from Exponent_Query import ExponentQuery
from EmbeddedStiffnessMatrix_Class import EmbeddedStiffnessMatrix 
from WriteInputfile_Class import WriteInputFile
from collections import defaultdict
from Xi_gamma_computer import XiGammaComputer
from Compute_velocity_mapping import VelocityMappingComputer, MuGammaCalculator
from Compute_linear_operators import LinearOperatorCache
from Compute_higher_order_terms import HigherOrderSolver
from Homological_equation_solver import HomologicalEquationSolver_normal_style
from Homological_equation_solver import HomologicalEquationSolver_graph_style
from Results_printer import ResultPrinter
from Resonance_handler import ResonanceHandler
from OrderOnesolver import OrderOneSolver

class Initializer:
    """Handles initialization of all data structures and computational components for DPIM."""
    
    def __init__(self, dpim):
        """Initialize with a DPIM instance to configure.
        
        Args:
            dpim: The SecondOrderDPIM instance to initialize
        """
        self.dpim = dpim

    def _initialize_multiindex_generator(self):
        """Initialize the multi-index generator for tracking nonlinear terms."""
        self.dpim.n_total = self.dpim.n_modes + self.dpim.num_forcing
        self.dpim.mexp = MultiExponentGenerator(
            num_variables=self.dpim.n_total,
            num_forcing=self.dpim.num_forcing,
            max_order=self.dpim.max_order,
            max_order_modes=self.dpim.max_order_modes,
            maximum_order_forcing=self.dpim.maximum_order_forcing
        )
        self.dpim.mexp.build_tables()  # Build exponent tables for all nonlinear terms
        print("Exponent Combinations:\n", self.dpim.mexp.exponent_table)
        print("\nOrder Information:\n", self.dpim.mexp.order_table)
        print("\ngenerator.exponent_dict:\n", self.dpim.mexp.exponent_dict)
        

    def _initialize_data_structures(self):
        """Initialize core data structures for storing results and querying terms."""
        self.dpim.query = ExponentQuery(
            self.dpim.mexp.exponent_table,
            self.dpim.mexp.order_table,
            self.dpim.mexp.exponent_dict, 
            self.dpim.n_modes,
            self.dpim.num_forcing
        )
        # Initialize containers for resonance tracking
        self.dpim.resonance_relations = defaultdict(list)
        self.dpim.resonant_terms = set()
        self.dpim.W_u_dict = {}  # Dictionary for nonlinear mappings
        self.dpim.velocity_computer = None  # Will be initialized later
        self.dpim.linear_operator_cache = None  # Cache for linear operators
        # Create human-readable column headers
        self.dpim.column_headers = [str(tuple(exp)) for exp in self.dpim.mexp.exponent_table['exponents']]


    def _initialize_eigenquantities(self):
        """Compute eigenquantities (modes, eigenvalues) for the linear system."""
        shift = 0.1  # Small shift for numerical stability
        self.dpim.Phi_i, self.dpim.lambda_i, self.dpim.Psi_i = eigensolutions_damped(
            K=self.dpim.K, 
            M=self.dpim.M, 
            sigma=shift, 
            a=self.dpim.a, 
            b=self.dpim.b, 
            k=self.dpim.n_modes + 10  # Compute extra modes for numerical stability
        )

        """import scipy.io

        # Load eigenvalues
        eigenvalues_data = scipy.io.loadmat('Cantilever_Beam_200_elements_eigenvalues.mat')
        self.dpim.lambda_i = eigenvalues_data['lambda_i'].flatten()  # Shape (12,)
        print(f"Eigenvalues shape: {self.dpim.lambda_i}")

        # Load eigenvectors
        eigenvectors_data = scipy.io.loadmat('Cantilever_Beam_200_elements_eigenvectors.mat')
        self.dpim.Phi_i = eigenvectors_data['phi']  # Shape (1800, 12)
        self.dpim.Psi_i = eigenvectors_data['psi']  # Shape (1800, 12)"""

        print(f"Phi shape: {self.dpim.Phi_i}")
        print(f"Psi shape: {self.dpim.Psi_i}")
        for i in range(len(self.dpim.lambda_i)):
            print(f"Phi dpim: {self.dpim.Phi_i[:, i]}")
            print(f"lambda dpim: {self.dpim.lambda_i[i]}")
            print(f"line 80 main dpim: {np.linalg.norm(self.dpim.lambda_i[i]**2 * (M @ self.dpim.Phi_i[:, i]) +C@ self.dpim.Phi_i[:, i]*self.dpim.lambda_i[i] + K @ self.dpim.Phi_i[:, i])/np.linalg.norm(K @ self.dpim.Phi_i[:, i])}")

    def _setup_modal_matrices(self):
        """Organize computed modes into primary and outer sets."""
        # Split modes into primary set (for parameterization) and outer set
        self.dpim.Phi = np.concatenate((
            self.dpim.Phi_i[:, :self.dpim.n_modes//2],
            self.dpim.Phi_i[:, self.dpim.n_modes//2+5:self.dpim.n_modes+5]
        ), axis=1)
        
        self.dpim.Psi = np.concatenate((
            self.dpim.Psi_i[:, :self.dpim.n_modes//2],
            self.dpim.Psi_i[:, self.dpim.n_modes//2+5:self.dpim.n_modes+5]
        ), axis=1)
        # Organize eigenvalues
        self.dpim.lambda_ = np.append(
            self.dpim.lambda_i[:self.dpim.n_modes//2], 
            self.dpim.lambda_i[self.dpim.n_modes//2+5:self.dpim.n_modes+5]
        )
   
        print(f"self.dpim.lambda_ :\n{self.dpim.lambda_ }")
        self.dpim.lambda_outer = np.append(
            self.dpim.lambda_i[self.dpim.n_modes//2:5+self.dpim.n_modes//2],
            self.dpim.lambda_i[self.dpim.n_modes+5:]
        )

    def _initialize_force_vectors(self):
        """Initialize forcing vectors in modal coordinates."""
        self.dpim.force_vectors = np.zeros((self.dpim.M.shape[0], self.dpim.n_forcing), dtype=complex)
        #Initialize first forcing vector as perturbation of first mode
        #self.dpim.force_vectors[:,0] = 0.5*self.dpim.amplitude_forcing* self.dpim.M@self.dpim.Phi_i[:, 0]
        #print(f"Force vector: {self.dpim.force_vectors}")

    def _initialize_lambda_tilde(self):
        """Initialize extended eigenvalue array including forcing frequencies."""
        self.dpim.lambda_tilde = np.concatenate((
            self.dpim.lambda_,
            [1j *w for w in self.dpim.frequency_forcing],  # Positive forcing frequencies
            [-1j*w for w in self.dpim.frequency_forcing]   # Negative forcing frequencies
        ))
        print(self.dpim.lambda_tilde)

    def _initialize_solution_matrices(self):
        """Initialize matrices for storing solution components."""
        n_sets = sum(self.dpim.mexp.order_table['count'])  # Total number of terms to compute
        # Initialize storage for nonlinear mappings
        self.dpim.Wu = np.zeros((self.dpim.n_dof, n_sets), dtype=np.complex128)  # Displacement mapping
        self.dpim.Wv = np.zeros((self.dpim.n_dof, n_sets), dtype=np.complex128)  # Velocity mapping
        self.dpim.f = np.zeros((self.dpim.n_modes, n_sets), dtype=np.complex128)  # Reduced dynamics coefficients

    def _initialize_computational_components(self):
        """Initialize specialized computational components."""
        # Component for computing RHS of homological equations
        self.dpim.xi_gamma_computer = XiGammaComputer(
            M=self.dpim.M, C=self.dpim.C, Wu=self.dpim.Wu, Wv=self.dpim.Wv, f=self.dpim.f,
            query=self.dpim.query, lambda_tilde=self.dpim.lambda_tilde, n_modes=self.dpim.n_modes
        )

        # Component for computing nonlinear corrections
        self.dpim.mu_gamma_computer = MuGammaCalculator(
            M=self.dpim.M, Phi=self.dpim.Phi, Wu=self.dpim.Wu, f=self.dpim.f,
            query=self.dpim.query
        )

        # Component for velocity mapping computations
        self.dpim.velocity_computer = VelocityMappingComputer(
            M=self.dpim.M, Phi=self.dpim.Phi, Wu=self.dpim.Wu, f=self.dpim.f,
            query=self.dpim.query, Wv=self.dpim.Wv
        )
        self.dpim.velocity_computer.lambda_ = self.dpim.lambda_
        
        # Cache for linear operators to avoid recomputation
        self.dpim.linear_operator_cache = LinearOperatorCache(
            M=self.dpim.M, C=self.dpim.C, K=self.dpim.K, Phi=self.dpim.Phi,
            lambda_=self.dpim.lambda_, n_modes=self.dpim.n_modes
        )

    def initialize(self):
        """Main initialization routine that coordinates all setup steps."""
        self._initialize_multiindex_generator()
        self._initialize_data_structures()
        try:
            self._initialize_eigenquantities()
            self._setup_modal_matrices()
            self._initialize_force_vectors()
            self._initialize_lambda_tilde()
            self._initialize_solution_matrices()
            self._initialize_computational_components()
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            raise


class SecondOrderDPIM:
    """Main class implementing Second Order Dynamic Parameterization of Invariant Manifolds."""
    
    def __init__(self, M, C, K, non_encased_nodes, frequency_forcing, 
                 max_order, max_order_modes, maximum_order_forcing,  
                 n_modes, n_forcing, amplitude_forcing, minimum_distance, resonance_tol, a, b,
                 base_input_filename):
        """Initialize the DPIM solver with system matrices and parameters.
        
        Args:
            M: Mass matrix
            C: Damping matrix  
            K: Stiffness matrix
            non_encased_nodes: List of non-embedded nodes
            frequency_forcing: Forcing frequencies
            max_order: Maximum order of nonlinearity to consider
            max_order_modes: Maximum order for modal terms
            maximum_order_forcing: Maximum order for forcing terms
            n_modes: Number of modes to include
            n_forcing: Number of forcing terms
            resonance_tol: Tolerance for resonance detection
            a, b: Parameters for eigenvalue computation
            base_input_filename: Base filename for output
        """
        # Store system matrices as sparse matrices
        self.M = csr_matrix(M)
        self.C = csr_matrix(C)
        self.K = csr_matrix(K)
        
        # Store parameters
        self.n_forcing = n_forcing
        self.max_order = max_order
        self.n_modes = n_modes * 2  # Include pairs of modes
        self.n_dof = M.shape[0]
        self.a = a
        self.b = b
        self.resonance_tol = resonance_tol
        self.base_input_filename = base_input_filename
        self.non_encased_nodes = non_encased_nodes
        self.frequency_forcing = frequency_forcing
        self.num_forcing = 2 * n_forcing  # Positive and negative frequencies
        self.maximum_order_forcing = maximum_order_forcing
        self.max_order_modes = max_order_modes
        self.amplitude_forcing=amplitude_forcing
        self.minimum_distance=minimum_distance
        # Initialize components
        self.initializer = Initializer(self)
        self.resonance_handler = ResonanceHandler(self)
        self.order_one_solver = OrderOneSolver(self)
        self.result_printer = ResultPrinter(self)
        
        # Perform initialization
        self.initializer.initialize()
        self.resonance_handler.detect_outer_resonances()

    def _detect_resonances(self, gamma, s):
        """Detect resonances for a specific term (delegates to ResonanceHandler)."""
        return self.resonance_handler._detect_resonances(gamma, s)

    def solve_higher_orders(self):
        """Solve homological equations for orders 2 through max_order."""
        print("\n=== Solving higher orders ===")
        higher_order_solver = HigherOrderSolver(self)
        for order in range(2, self.max_order + 1):
            print(f"\n=== Processing order {order} ===")
            higher_order_solver.process_order(order, self.n_modes)

    def compute_parametrization(self):
        """Main driver for computing the complete parameterization."""
        # Solve first order terms
        self.order_one_solver.solve_order_one()
        
        # Solve higher order terms if needed
        if self.max_order > 1:
            self.solve_higher_orders()
            
        # Save and print results
        self.result_printer._save_results_to_file()
        df_wu, df_wv, df_f = self.result_printer._create_result_dataframes()
        self.result_printer._print_final_results(df_wu, df_wv, df_f)
        self.result_printer.save_to_txt("output_for_FRF.txt")
        self.result_printer.save_Wu_to_file("Wu_matrix.txt")
        self.result_printer.save_Wv_to_file("Wv_matrix.txt")

    def _get_linear_operator(self, s):
        """Get linear operator for frequency s (uses caching)."""
        return self.linear_operator_cache.get_operator(s)

    def _compute_xi_gamma(self, gamma, i_gl):
        """Compute RHS vector Ξ_γ for term γ."""
        return self.xi_gamma_computer.compute_xi_gamma(gamma, i_gl)

    def _compute_mu_gamma(self, gamma):
        """Compute nonlinear correction term μ_γ for term γ.""" 
        return self.mu_gamma_computer.compute_mu_gamma(gamma)

    def _compute_velocity_mapping(self, gamma, i_gl, mu_gamma):
        """Compute velocity mapping Wv for term γ."""
        self.velocity_computer.lambda_tilde = self.lambda_tilde
        self.Wv[:, i_gl-1] = self.velocity_computer.compute_velocity_mapping(gamma, i_gl, mu_gamma)

    def _solve_homological_equation(self, gamma, s, rhs, i_gl, mu_gamma, is_resonant=False, resonant_modes=None):
        """Solve the homological equation for term γ (delegates to solver)."""

        #if not hasattr(self, 'homological_solver'):
        #    self.homological_solver = HomologicalEquationSolver_graph_style(self)
        #return self.homological_solver.solve(gamma, s, rhs, i_gl, self.n_modes, mu_gamma)

        if not hasattr(self, 'homological_solver'):
            self.homological_solver = HomologicalEquationSolver_normal_style(self)
        return self.homological_solver.solve(gamma, s, rhs, i_gl, is_resonant, resonant_modes, mu_gamma, )  



if __name__ == "__main__":
    #####TIRAR FORCE VECTOR COMENTÁRIO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ##### No compute_higher_order_terms NÃO ESQUECER DE TIRAR *0 EM G E H

    start = time.perf_counter()
    #####################Change by user#############################
    n_modes=1
    minimum_distance=10 #minimum distance between two points (mesh) from your model #20.0 MEMS #0.1 # CANTILEVER 1/10
    max_order=1
    max_order_modes=1
    maximum_order_forcing=1
    resonance_tol=1e-3
    amplitude_forcing=2e-3
    a, b = 0.5e-6, 0.0  # Small damping (effectively undamped)

    with open('Homological_equation.txt', 'w') as file:
        file.write("")  # Writes an empty string to clear the file

    with open('rhs_equation.txt', 'w') as file:
        file.write("")  # Writes an empty string to clear the file


    #MEMES
    frequency_forcing = [1.819006] #1.819006
    input_file = "mems_abaqus_UEL_modified.inp"
    mass_matrix = "Memes_240_elements_mass_matrix.mtx"
    linear_stiffeness_matrix="Memes_240_elements_stiffness_matrix.mtx"
    base_input_filename="mems_abaqus_UEL_modified"

    #Clamped clamped Beam with 280 elements
    #frequency_forcing = [] #33.355528
    #input_file = "Clamped_clamped_beam_280_elements_UEL_modified.inp"
    #mass_matrix = "Clamped_clamped_Beam_280_elements_mass_matrix.mtx"
    #linear_stiffeness_matrix="Clamped_clamped_Beam_280_elements_stiffness_matrix.mtx"
    #base_input_filename="Clamped_clamped_beam_280_elements_UEL_modified"

    #Cantilever 500 elements
    #frequency_forcing = [] #8.425818
    #input_file = "Cantilever_Beam_500_elements_UEL_modified.inp"
    #mass_matrix = "Cantilever_Beam_500_elements_mass_matrix.mtx"
    #linear_stiffeness_matrix="Cantilever_Beam_500_elements_stiffness_matrix.mtx"
    #base_input_filename="Cantilever_Beam_500_elements_UEL_modified"

    #Cantilever 2000 elements
    #frequency_forcing = [8.425818] #33.355528
    #input_file = "Cantilever_Beam_2000_elements_UEL_modified.inp"
    #mass_matrix = "Cantilever_Beam_2000_elements_stiffness_matrix.mtx"
    #linear_stiffeness_matrix="Cantilever_Beam_2000_elements_mass_matrix.mtx"
    #base_input_filename="Cantilever_Beam_2000_elements_UEL_modified"

    #CANTILIVER BEAM 87 ELEMENTS
    #frequency_forcing = [150.848415] #150.848415
    #input_file = "Cantiliver_Beam_87_elements_modified.inp"
    #mass_matrix = "Cantiliver_Beam_87_elements_mass_matrix.mtx"
    #linear_stiffeness_matrix="Cantiliver_Beam_87_elements_stiffness_matrix.mtx"
    #base_input_filename="Cantiliver_Beam_87_elements_modified"

    #Cantilever Beam with 200 elements
    #frequency_forcing = [6.517843] #33.355528
    #input_file = "Cantilever_beam_200_elements_UEL_modified.inp"
    #mass_matrix = "Cantilever_Beam_200_elements_mass_matrix.mtx"
    #linear_stiffeness_matrix="Cantilever_Beam_200_elements_stiffness_matrix.mtx"
    #base_input_filename="Cantilever_beam_200_elements_UEL_modified"

    #SINGLE ELEMENT
    #frequency_forcing = []  #3.23128305 #3231.283054
    #input_file = "Meu_input_file.inp"
    #mass_matrix = "Meu_input_file_mass_matrix_E_210e9.mtx"
    #linear_stiffeness_matrix="Meu_input_file_stiffness_matrix_E_210e9.mtx"
    #base_input_filename="Meu_input_file"


    #TWO ELEMENTS
    #frequency_forcing = [1.47210157j]
    #input_file = "Two_elements_UEL.inp"
    #mass_matrix = "Two_elements_mass_matrix.mtx"
    #linear_stiffeness_matrix="Two_elements_stiffness_matrix.mtx"
    #base_input_filename="Two_elements_UEL"

    #CANTILIVER BEAM 20 ELEMENTS
    #frequency_forcing = [0.06521j]
    #input_file = "twenty_elements_Cantiliver_Beam_UEL.inp"
    #mass_matrix = "twenty_elements_mass_matrix.mtx"
    #linear_stiffeness_matrix="twenty_elements_stiffness_matrix.mtx"
    #base_input_filename="twenty_elements_Cantiliver_Beam_UEL"


    #CLAMPED CLAMPED BEAM 60 ELEMENTS
    #frequency_forcing = [724.161876]
    #input_file = "Clamped_clamped_Beam_60_elements_modified.inp"
    #mass_matrix = "Clamped_clamped_60_elements_mass_matrix.mtx"
    #linear_stiffeness_matrix="Clamped_clamped_60_elements_stiffness_matrix.mtx"
    #base_input_filename="Clamped_clamped_Beam_60_elements_modified"

    #CLAMPED CLAMPED BEAM 60 ELEMENTS
    #frequency_forcing = [41.498247]
    #input_file = "Clamped_Clamped_beam_200_elements_UEL_modified.inp"
    #mass_matrix = "Clamped_clamped_Beam_200_elements_mass_matrix.mtx"
    #linear_stiffeness_matrix="Clamped_clamped_Beam_200_elements_stiffness_matrix.mtx"
    #base_input_filename="Clamped_Clamped_beam_200_elements_UEL_modified"

    #CANTILIVER BEAM 120 ELEMENTS
    #frequency_forcing = [18.115095j]
    #input_file = "120_elements_Cantiliver_Beam_UEL_modified.inp"
    #mass_matrix = "Cantiliver_Beam_120_elements_mass_matrix.mtx"
    #linear_stiffeness_matrix="Cantiliver_Beam_120_elements_stiffness_matrix.mtx"
    #base_input_filename="120_elements_Cantiliver_Beam_UEL_modified"


    #SINGLE STRANGE ELEMENT
    #frequency_forcing = [3.43718279j]
    #input_file = "Meu_input_file_strange_cube.inp"
    #mass_matrix = "Meu_input_file_strange_cube_mass_matrix.mtx"
    #linear_stiffeness_matrix="Meu_input_file_strange_cube_stiffness_matrix.mtx"
    #base_input_filename="Meu_input_file_strange_cube"

    #BEAM CLAMPED-CLAMPED
    #frequency_forcing = []#[713.042993j] # One forcing frequencies
    #input_file = "UEL_600_elements_modified.inp"
    #mass_matrix = "UEL_600_elements_modified_mass_matrix.mtx"
    #linear_stiffeness_matrix="UEL_600_elements_stiffness_matrix.mtx"
    #base_input_filename="UEL_600_elements_modified"
     ##################### ##################### #####################

    file = WriteInputFile(input_file)
    total_nodes, encased_nodes, non_encased_nodes = file.get_node_information()
    print(f"Total nodes: {total_nodes}")
    print(f"encased_nodes: {encased_nodes}")

    n_forcing = len(frequency_forcing)
    
    stiffness_matrix=EmbeddedStiffnessMatrix(non_encased_nodes, linear_stiffeness_matrix)
    K=stiffness_matrix.load_and_extract_non_embedded_matrix()
    print(f"K matrix:\n {K.shape}")
    mass_matrix=EmbeddedStiffnessMatrix(non_encased_nodes, mass_matrix)
    M=mass_matrix.load_and_extract_non_embedded_matrix()  
    print(f"M matrix:\n {M.shape}")
    C = a * M + b * K  # Element-wise scalar multiplication and addition

    mmwrite("Memes_240_elements_mass_matrix_sparse.mtx", M)
    mmwrite("Memes_240_elements_stiffness_matrix_sparse.mtx", K)
    import scipy.io as sio
    from scipy.io import mmread

    # Read the Matrix Market file (preserves sparsity)
    matrix_M = mmread("Memes_240_elements_mass_matrix_sparse.mtx")
    matrix_K = mmread("Memes_240_elements_stiffness_matrix_sparse.mtx")

    # Save as .mat file with a specific variable name
    sio.savemat("Memes_240_elements_mass_matrix_sparse.mat", {'M': matrix_M})
    sio.savemat("Memes_240_elements_stiffness_matrix_sparse.mat", {'K': matrix_K})

    print("Conversion completed successfully!")


    ################Change by the user###############################
    "Define force vectors"
    # _initialize_force_vectors()
    "Clamped clamped beam"
    #Change WriteInputfile_Class to WriteInputfile_Class_clamped_clamped
    #Change Duplicate input file to Duplicate_Input_File_Clamped_Clamped
     ##################### ##################### ##################### 
    

    dpim = SecondOrderDPIM(
        M, C, K, non_encased_nodes, 
        frequency_forcing,
        max_order=max_order, max_order_modes=max_order_modes, maximum_order_forcing=maximum_order_forcing, n_modes=n_modes, n_forcing=n_forcing, amplitude_forcing=amplitude_forcing, minimum_distance=minimum_distance, resonance_tol=resonance_tol, a=a, b=b,
        base_input_filename=base_input_filename
    )
    dpim.compute_parametrization()

    end = time.perf_counter()
    print(f"Time taken: {end - start:.6f} seconds")


