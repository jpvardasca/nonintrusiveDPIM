import numpy as np
import re


class InputFileModifier:
    """Modifies Abaqus input files to apply specified displacements and update output filenames.
    
    This class handles:
    - Reading and parsing Abaqus input files
    - Extracting node boundary information
    - Applying new displacement boundary conditions
    - Updating output matrix filenames
    - Writing modified input files
    """

    def __init__(self, input_filename, displacement, output_filename):
        """
        Initialize the input file modifier.
        
        Args:
            input_filename (str): Path to original Abaqus input file
            displacement (list): List of displacement values to apply (flattened [u1, u2, u3] per node)
            output_filename (str): Desired output filename for modified input file
        """
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.displacement =displacement
        self.header, self.original_boundary, self.footer = self._read_and_split_file()
        self.node_order = self._extract_node_order()

    def _read_and_split_file(self):
        """
        Read and split the input file into logical sections.
        
        Returns:
            tuple: (header, boundary_section, footer) strings
        """
        with open(self.input_filename, 'r') as f:
            content = f.read()
        boundary_start = "** Boundary conditions"
        boundary_end = "** Output requests"
        parts = content.split(boundary_start)
        boundary_parts = parts[1].split(boundary_end)
        return parts[0] + boundary_start, boundary_parts[0], boundary_end + boundary_parts[1]

    def _extract_node_order(self):
        """
        Extract and validate the node ordering from boundary conditions.
        
        Returns:
            list: Ordered list of node names (e.g., ['node1', 'node2', ...])
            
        Raises:
            ValueError: If no node boundary conditions are found
        """
        matches = re.findall(r'^(node\d+),\s+\d,\s+\d', self.original_boundary, re.MULTILINE)
        node_order, seen_nodes = [], set()
        for node in matches:
            if node not in seen_nodes:
                seen_nodes.add(node)
                node_order.append(node)
        if not node_order:
            raise ValueError("No node boundary conditions found in the input file")
        return node_order

    def _create_boundary_section(self, displacements):
        """
        Create new boundary condition section with specified displacements.
        
        Args:
            displacements (list): Flattened list of displacement values
            
        Returns:
            str: Formatted boundary condition section
        """
        new_boundary = "\n*Boundary\nSet-2, ENCASTRE\n"
        for i, node in enumerate(self.node_order):
            u1, u2, u3 = displacements[3*i: 3*i + 3]
            new_boundary += f"{node}, 1, 1, {u1}\n{node}, 2, 2, {u2}\n{node}, 3, 3, {u3}\n"
        return new_boundary

    def _update_matrix_filename(self, footer, new_filename):
        """
        Update the matrix output filename in the footer section.
        
        Args:
            footer (str): Original footer content
            new_filename (str): New filename (without extension)
            
        Returns:
            str: Updated footer content
        """
        # Find and replace the File Name parameter in the *Element Matrix Output line
        pattern = r'(\*Element Matrix Output, Elset=allUEL, File Name=)([^,]+)(, Output File=User Defined, Stiffness=Yes)'
        replacement = fr'\1{new_filename}\3'
        updated_footer = re.sub(pattern, replacement, footer)
        return updated_footer

    def create_modified_input_files(self):
        """
        Create modified input file with new boundary conditions and updated output filename.
        
        Raises:
            ValueError: If displacement data is insufficient for all nodes
        """
        if len(self.displacement) < 3 * len(self.node_order):
            raise ValueError("Displacement has insufficient data.")
        
        matrix_filename = self.output_filename.replace('.inp', '')  # Remove extension for matrix filename
        
        # Update the footer with the new matrix filename
        updated_footer = self._update_matrix_filename(self.footer, matrix_filename)
        
        new_content = self.header + self._create_boundary_section(self.displacement) + updated_footer
        with open(self.input_filename, 'w') as f:
            f.write(new_content)
        print(f"Created new {self.input_filename}")

