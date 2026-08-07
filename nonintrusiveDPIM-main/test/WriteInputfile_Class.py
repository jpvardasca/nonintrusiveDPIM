class WriteInputFile:
    def __init__(self, input_file: str):
        """
        Initialize NodeReader class to read node information from an Abaqus input file.
        
        Args:
            input_file: Path to the .inp file
        """
        self.input_file = input_file
        self.lines = self._read_file()
        self.total_nodes = None
        self.encased_nodes = set()
        self.non_encased_nodes = set()

    def _read_file(self):
        """Read the input file and store its lines."""
        with open(self.input_file, 'r') as file:
            return file.readlines()

    def get_node_information(self):
        """
        Extract node information from the input file.
        
        Returns:
            tuple: (total_nodes, encased_nodes, non_encased_nodes)
        """
        self._find_total_nodes()
        self._find_encased_nodes()
        self._calculate_non_encased_nodes()
        return self.total_nodes, self.encased_nodes, self.non_encased_nodes

    def _find_total_nodes(self):
        """Find the total number of nodes from the *Node section."""
        node_count = 0
        in_node_section = False
        
        for line in self.lines:
            if line.startswith('*Node'):
                in_node_section = True
                continue
            if in_node_section:
                if line.strip() == '' or line.startswith('*'):
                    break
                if line.split(',')[0].strip().isdigit():
                    node_count += 1
        
        self.total_nodes = node_count

    def _find_encased_nodes(self):
        """Find nodes in the encased set (Set-2)."""
        in_encased_set = False
        
        for line in self.lines:
            if line.strip().startswith('*Nset, nset=Set-2'):
                in_encased_set = True
                continue
            if in_encased_set:
                if line.strip() == '' or line.startswith('*'):
                    break
                # Extract node numbers from the line
                nodes = [x.strip() for x in line.split(',') if x.strip().isdigit()]
                self.encased_nodes.update(map(int, nodes))

    def _calculate_non_encased_nodes(self):
        """Calculate non-encased nodes by subtracting encased nodes from total."""
        if self.total_nodes is None:
            self._find_total_nodes()
        if not self.encased_nodes:
            self._find_encased_nodes()
        
        all_nodes = set(range(1, self.total_nodes + 1))
        self.non_encased_nodes = all_nodes - self.encased_nodes


# Example usage:
#if __name__ == "__main__":
   # reader = WriteInputFile("Meu_input_file_1.inp")
   # total_nodes = reader.get_node_information()
    
   # print(f"Total nodes: {total_nodes}")