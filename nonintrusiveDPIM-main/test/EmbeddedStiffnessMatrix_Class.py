import numpy as np
from scipy.sparse import csr_matrix

class EmbeddedStiffnessMatrix:
    def __init__(self, non_embedded_nodes, file_name):
        """Initialize the EmbeddedStiffnessMatrix object.
        
        Args:
            non_embedded_nodes: List of nodes that are not embedded (free nodes)
            file_name: Name of the file containing the stiffness matrix data
        """
        self.non_embedded_nodes = non_embedded_nodes
        self.file_name = file_name
        self.non_embedded_dofs = self._calculate_dofs()  # Calculate DOFs for non-embedded nodes
        self.dof_to_index = {dof: idx for idx, dof in enumerate(self.non_embedded_dofs)}  # Create mapping from DOF to index
        self.reduced_size = len(self.non_embedded_dofs)  # Size of the reduced stiffness matrix

    def _calculate_dofs(self):
        """Calculate the degrees of freedom (DOFs) for non-embedded nodes.
        
        Returns:
            List of DOFs where each node contributes 3 DOFs (x, y, z displacements)
        """
        return [dof for node in self.non_embedded_nodes 
                for dof in (node*3-2, node*3-1, node*3)]

    def _remap_dofs(self, row, col, val):
        """Remap the DOFs to indices in the reduced stiffness matrix.
        
        Args:
            row: Row indices from the full stiffness matrix
            col: Column indices from the full stiffness matrix
            val: Values from the full stiffness matrix
            
        Returns:
            Filtered and remapped row indices, column indices, and values
            for the reduced stiffness matrix
        """
        # Filter for entries where both row and col are in non-embedded DOFs
        mask = np.isin(row, self.non_embedded_dofs) & np.isin(col, self.non_embedded_dofs)
        f_row, f_col, f_val = row[mask], col[mask], val[mask]
        
        # Remap DOFs to reduced matrix indices
        f_row = np.vectorize(self.dof_to_index.get)(f_row)
        f_col = np.vectorize(self.dof_to_index.get)(f_col)
        return f_row, f_col, f_val

    def load_and_extract_non_embedded_matrix(self):
        """Load the stiffness matrix from file and extract the non-embedded portion.
        
        Returns:
            A compressed sparse row (CSR) matrix containing only the non-embedded portion
            of the stiffness matrix
            
        Raises:
            FileNotFoundError: If the specified file cannot be loaded
        """
        try:
            # Load data from file (assumed to be in row, column, value format)
            Data = np.loadtxt(self.file_name)
            row, col, val = Data[:,0].astype(int), Data[:,1].astype(int), Data[:,2]
        except Exception as e:
            raise FileNotFoundError(f"FEM: In function BuildSparseMatrix, parameter 'fileName' is invalid. "
                                    f"\033[91m{f'filename = {self.file_name}'}\033[0m") from e
        
        # Remap DOFs and create sparse matrix
        f_row, f_col, f_val = self._remap_dofs(row, col, val)
        return csr_matrix((f_val, (f_row, f_col)), 
                         shape=(self.reduced_size, self.reduced_size))
    
if __name__ == "__main__":
    
    non_embedded_nodes = []  # List of nodes that are not embedded
    #number_max_vectors = 3         # Number of mode vectors to consider

    
    #run_abaqus_job('Linear_Beam_modified_check.inp')

    stiffness_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'mems_abaqus_UEL_modified_0.mtx')
    rudy=stiffness_matrix.load_and_extract_non_embedded_matrix()

    stiffness_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'mems_abaqus_UEL_modified_1.mtx')
    rudy=stiffness_matrix.load_and_extract_non_embedded_matrix()

    #mmwrite('Eigen_matrix.mtx', csr_matrix(rudy))

    #dofs=stiffness_matrix._calculate_dofs()
    #print(f"dofs: {dofs}")
    #K=stiffness_matrix.load_and_extract_non_embedded_matrix()
    
    #print(f"matrix: {K}")
    
    