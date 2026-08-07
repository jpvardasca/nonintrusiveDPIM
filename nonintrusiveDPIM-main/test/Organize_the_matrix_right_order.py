import re
import numpy as np
from collections import defaultdict
from decimal import Decimal, getcontext
import os

class MatrixProcessor:
    def __init__(self, input_file):  # Removed output_file parameter
        """Initialize the matrix processor with input file path.
        
        Args:
            input_file: Path to the Abaqus matrix output file
        """
        self.input_file = input_file
        self.elements = []
        self.is_symmetric = True
        getcontext().prec = 28
    
    def determine_symmetry(self):
        """Determine matrix symmetry by scanning for 'UNSYM' keyword in USER ELEMENT line."""
        try:
            with open(self.input_file, 'r') as f:
                for line in f:
                    if '*USER ELEMENT' in line and 'UNSYM' in line:
                        self.is_symmetric = False
                        break
        except Exception as e:
            print(f"Warning: Could not determine matrix symmetry. Defaulting to symmetric. Error: {str(e)}")
    
    def read_and_parse_file(self):
        """Main file parsing driver that determines symmetry and routes to appropriate parser."""
        print(f"\nReading {self.input_file}")
        self.determine_symmetry()
        print(f"Matrix type: {'Symmetric' if self.is_symmetric else 'Non-symmetric'}")
        
        if self.is_symmetric:
            self._read_symmetric()
        else:
            self._read_general()
    
    def _read_symmetric(self):
        """Parse symmetric matrix format from Abaqus output file."""
        with open(self.input_file, 'r') as f:
            content = f.read()
        
        element_blocks = re.split(r'\*\* ELEMENT NUMBER\s+\d+ STEP NUMBER\s+\d+ INCREMENT NUMBER\s+\d+', content)[1:]
        
        for block in element_blocks:
            node_lines = []
            for line in block.split('\n'):
                if 'ELEMENT NODES' in line or ('**' in line and any(str(i) in line for i in range(1, 100))):
                    node_lines.append(line)
            
            nodes = []
            for line in node_lines:
                if '**' in line and ',' in line:
                    nums = re.findall(r'\d+', line.split('**')[1])
                    nodes.extend([int(n.strip()) for n in nums if n.strip()])
            
            matrix_lines = []
            in_matrix = False
            for line in block.split('\n'):
                line = line.strip()
                if '*MATRIX' in line:
                    in_matrix = True
                    continue
                if in_matrix and line and not line.startswith('*'):
                    if not re.match(r'^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*$', line):
                        matrix_lines.append(line)
            
            matrix_data = []
            for line in matrix_lines:
                parts = re.split(r'[,\s]+', line.strip())
                for num in parts:
                    if num:
                        try:
                            matrix_data.append(float(num))
                        except ValueError:
                            continue
            
            expected_values = 24 * 25 // 2
            if len(matrix_data) != expected_values:
                print(f"Warning: Expected {expected_values} values, got {len(matrix_data)}")
                continue
            
            mat_size = 24
            k_element = np.zeros((mat_size, mat_size))
            idx = 0
            for i in range(mat_size):
                for j in range(i + 1):
                    if idx < len(matrix_data):
                        k_element[i, j] = matrix_data[idx]
                        k_element[j, i] = matrix_data[idx]
                        idx += 1
            
            self.elements.append({
                'nodes': nodes,
                'matrix': k_element
            })
    
    def _read_general(self):
        """Parse non-symmetric matrix format from Abaqus output file."""
        current = None
        state = None
        
        with open(self.input_file) as f:
            for line in (l.strip() for l in f if l.strip()):
                if line.startswith('** ELEMENT NUMBER'):
                    if current:
                        self._process_element(current)
                    current = {
                        'number': int(re.search(r'\d+', line).group()),
                        'nodes': [], 'dofs': [], 'matrix': None,
                        'data': []
                    }
                    state = None
                elif not current:
                    continue
                elif '** ELEMENT NODES' in line:
                    state = 'nodes'
                elif '*MATRIX' in line:
                    state = 'matrix'
                elif state == 'nodes':
                    current['nodes'] = [int(n) for n in re.sub(r'\*\*', '', line).split(',') if n]
                    state = 'dofs'
                elif state == 'dofs':
                    current['dofs'] = [int(d) for d in re.sub(r'\*\*', '', line).split(',') if d]
                    state = None
                elif state == 'matrix':
                    numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', line)
                    current['data'].extend([Decimal(num) for num in numbers])
        
        if current:
            self._process_element(current)
    
    def _process_element(self, element):
        """Convert raw matrix data into numpy array for a non-symmetric element.
        
        Args:
            element: Element dictionary with raw data
        """
        size = len(element['nodes']) * len(element['dofs'])
        expected = size ** 2
        if len(element['data']) == expected:
            try:
                element['matrix'] = np.array([float(d) for d in element['data']]).reshape((size, size))
            except ValueError:
                element['matrix'] = None
        del element['data']
        self.elements.append(element)
    
    def assemble_global_matrix(self):
        """Assemble global matrix from element contributions.
        
        Returns:
            Global stiffness matrix (symmetric) or DOF mapping (non-symmetric)
        """
        if not self.elements:
            return None
        
        if self.is_symmetric:
            return self._assemble_symmetric()
        else:
            return self._assemble_general()
    
    def _assemble_symmetric(self):
        """Assemble global symmetric stiffness matrix."""
        all_nodes = set()
        for elem in self.elements:
            all_nodes.update(elem['nodes'])
        
        if not all_nodes:
            raise ValueError("No nodes found in elements")
        
        sorted_nodes = sorted(all_nodes)
        max_node = sorted_nodes[-1]
        dof_per_node = 3
        global_size = max_node * dof_per_node
        
        node_to_dof = {node: (node-1)*dof_per_node for node in sorted_nodes}
        K_global = np.zeros((global_size, global_size))
        
        for elem in self.elements:
            nodes = elem['nodes']
            if len(nodes) != 8:
                print(f"Warning: Element has {len(nodes)} nodes (expected 8)")
                continue
                
            k_elem = elem['matrix']
            
            local_to_global = []
            for local_node_idx, node in enumerate(nodes):
                base_dof = node_to_dof[node]
                local_to_global.extend([
                    (node, 0, base_dof),
                    (node, 1, base_dof+1),
                    (node, 2, base_dof+2)
                ])
            
            original_dof_order = [dof for (node, local_dof, dof) in local_to_global]
            sorted_ltg = sorted(local_to_global, key=lambda x: x[0])
            sorted_dof_order = [dof for (node, local_dof, dof) in sorted_ltg]
            
            perm_matrix = np.zeros((24, 24))
            for i, dof in enumerate(original_dof_order):
                new_pos = sorted_dof_order.index(dof)
                perm_matrix[new_pos, i] = 1
            
            k_elem_ordered = perm_matrix @ k_elem @ perm_matrix.T
            dof_map = sorted_dof_order
            
            for i_local in range(24):
                i_global = dof_map[i_local]
                for j_local in range(24):
                    j_global = dof_map[j_local]
                    K_global[i_global, j_global] += k_elem_ordered[i_local, j_local]
        
        return K_global
    
    def _assemble_general(self):
        """Assemble non-symmetric matrix using coordinate format storage."""
        dofs = sorted(self.elements[0]['dofs'])
        nodes = sorted({n for e in self.elements for n in e['nodes']})
        node_dof_map = {(n, d): i for i, (n, d) in enumerate((n, d) for n in nodes for d in dofs)}
        matrix = defaultdict(Decimal)
        
        for elem in self.elements:
            if elem['matrix'] is None:
                continue
            lg_map = [node_dof_map[(n, d)] for n in elem['nodes'] for d in dofs]
            for i_local, i_global in enumerate(lg_map):
                for j_local, j_global in enumerate(lg_map):
                    val = Decimal(str(elem['matrix'][i_local, j_local]))
                    if abs(val) > Decimal('1e-20'):
                        matrix[(i_global, j_global)] += val
        return matrix
    
    def _save_symmetric(self, matrix, output_file):
        """Save symmetric matrix in coordinate format (only lower triangle).
        
        Args:
            matrix: Numpy matrix to save
            output_file: Destination file path
        """
        n_rows, n_cols = matrix.shape
        with open(output_file, 'w') as f:
            for i in range(n_rows):
                for j in range(i, n_cols):
                    val = matrix[i, j]
                    if abs(val) > 1e-12:
                        # Increase precision from 15 to 20 decimal places
                        line = f"{i+1} {j+1} {val:.20e}\n"
                        line = line.replace('e', 'E')  # Maintain capital E exponent
                        f.write(line)
    
    def _save_general(self, matrix, output_file):
        """Save non-symmetric matrix in coordinate format.
        
        Args:
            matrix: Dictionary of (row, col): value entries
            output_file: Destination file path
        """
        entries = []
        for (i, j), v in matrix.items():
            if abs(v) > Decimal('1e-20'):
                # Format Decimal directly without float conversion
                entries.append((i, j, v))
        
        entries.sort(key=lambda x: (x[0], x[1]))
        
        with open(output_file, 'w') as f:
            for i, j, v in entries:
                # Format Decimal with 20 decimal places directly
                line = f"{i+1} {j+1} {v:.20e}\n"
                line = line.replace('e', 'E')  # Maintain capital E exponent
                f.write(line)

    def process_matrix_file(self):
        """Main processing pipeline: read, assemble, and save global matrix."""
        print("="*50 + "\nMatrix Assembly\n" + "="*50)
        self.read_and_parse_file()
        if self.elements:
            matrix = self.assemble_global_matrix()
            # Create temporary file
            temp_output = self.input_file + ".tmp"
            if self.is_symmetric:
                self._save_symmetric(matrix, temp_output)
            else:
                self._save_general(matrix, temp_output)
            
            # Replace original file with temporary file
            if os.path.exists(self.input_file):
                os.remove(self.input_file)
            os.rename(temp_output, self.input_file)
            print(f"\nGlobal matrix written to {self.input_file}")
            
            if self.is_symmetric and isinstance(matrix, np.ndarray):
                print(f"Global matrix size: {matrix.shape[0]}x{matrix.shape[1]}")
                print(f"Number of non-zero entries: {np.count_nonzero(matrix)}")
        print("\nDone")
        



# Example usage
if __name__ == "__main__":
    # Create processor instance
    processor = MatrixProcessor('Cantilever_Beam_2000_elements_stiffness_matrix.mtx')
    
    # Process the matrix file
    processor.process_matrix_file()
    #print(f"Processed matrix saved to: {output_file}")