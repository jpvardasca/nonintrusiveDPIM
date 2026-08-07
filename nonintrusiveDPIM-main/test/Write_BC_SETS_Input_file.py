import sys

class WriteInputFile_nodes:

    def __init__(self, input_file: str):
        """
        Initialize WriteInputFile class. Modifies the input file to include 
        node sets and boundary conditions for all nodes.
        """
        self.input_file = input_file
        self.lines = self._read_file()
        self.sets_info = {
            'All': {'type': 'generate', 'nodes': set()},
            'Set-2': {'nodes': set()}
        }

    def _read_file(self):
        """Read the input file and store its lines."""
        with open(self.input_file, 'r') as file:
            return file.readlines()

    def _write_file(self, output_file: str):
        """Write the modified lines to the output file."""
        with open(output_file, 'w') as file:
            file.writelines(self.lines)

    def parse_node_sets(self):
        """Parse all relevant node sets from the input file."""
        current_set = None
        
        for idx, line in enumerate(self.lines):
            stripped_line = line.strip()
            if stripped_line.startswith('*'):
                current_set = None
                if stripped_line.startswith('*Nset'):
                    if 'nset=All' in stripped_line:
                        current_set = 'All'
                        if 'generate' in stripped_line:
                            self.sets_info['All']['type'] = 'generate'
                    elif 'nset=Set-2' in stripped_line:
                        current_set = 'Set-2'
            else:
                if current_set == 'All' and self.sets_info['All']['type'] == 'generate':
                    parts = stripped_line.replace(' ', '').split(',')
                    if len(parts) >= 3:
                        start = int(parts[0])
                        end = int(parts[1])
                        step = int(parts[2])
                        self.sets_info['All']['nodes'] = set(range(start, end + 1, step))
                elif current_set == 'Set-1' or current_set == 'Set-2':
                    parts = stripped_line.replace(' ', '').split(',')
                    for part in parts:
                        if part != '':
                            try:
                                node_id = int(part)
                                self.sets_info[current_set]['nodes'].add(node_id)
                            except ValueError:
                                pass

    def get_all_nodes(self):
        """Get all nodes from the All set."""
        return sorted(self.sets_info['All']['nodes'])

    def find_node_section_end(self):
        """Find the end of the node sets section (after Set-1)."""
        for idx, line in enumerate(self.lines):
            stripped_line = line.strip()
            if stripped_line.startswith('*Nset') and 'nset=Set-2' in stripped_line:
                j = idx + 1
                while j < len(self.lines) and not self.lines[j].strip().startswith('*'):
                    j += 1
                return j  # Return index after last line of Set-1
        return None

    def find_boundary_location(self):
        """Find where to insert new boundary conditions."""
        for idx, line in enumerate(self.lines):
            if line.strip().startswith('*Boundary'):
                # Find end of boundary section
                j = idx + 1
                while j < len(self.lines) and not self.lines[j].strip().startswith('*'):
                    j += 1
                return j  # Insert after existing boundary conditions
        return None

    def create_nsets(self, nodes):
        """Create new node set definitions."""
        nsets = []
        for node in nodes:
            nsets.append(f'*Nset, nset=node{node}\n')
            nsets.append(f'{node}\n')
        return nsets

    def create_boundary_conditions(self, nodes):
        """Create boundary condition lines for nodes not in Set-2."""
        set1_nodes = self.sets_info['Set-2']['nodes']
        bcs = []
        for node in nodes:
            if node not in set1_nodes:
                for dof in [1, 2, 3]:
                    bcs.append(f'node{node}, {dof}, {dof}, 0.0\n')
        return bcs

    def modify_file(self, output_file):
        """Main method to modify the input file."""
        self.parse_node_sets()
        all_nodes = self.get_all_nodes()
        
        if not all_nodes:
            print("No nodes found")
            return
            
        # Create new node sets
        nsets = self.create_nsets(all_nodes)
        
        # Find insertion points
        node_section_end = self.find_node_section_end()
        bc_location = self.find_boundary_location()
        
        # Insert node sets after last node set definition
        if node_section_end is not None:
            self.lines = self.lines[:node_section_end] + nsets + self.lines[node_section_end:]
            
            # Adjust boundary location index if needed
            if bc_location is not None and bc_location >= node_section_end:
                bc_location += len(nsets)
        else:
            print("Warning: Could not find node sets section. Node sets not added.")
        
        # Create and insert boundary conditions
        if bc_location is not None:
            bcs = self.create_boundary_conditions(all_nodes)
            self.lines = self.lines[:bc_location] + bcs + self.lines[bc_location:]
        else:
            print("Warning: *Boundary section not found. Boundary conditions not added.")
        
        self._write_file(output_file)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Use command-line arguments if provided
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # Use default filenames if no arguments provided
        input_file = "Cantilever_Beam_2000_elements_UEL.inp"
        output_file = "Cantilever_Beam_2000_elements_UEL_modified.inp"
    
    modifier = WriteInputFile_nodes(input_file)
    modifier.modify_file(output_file)
    print(f"Modified file saved as {output_file}")