import os
import numpy as np
from Organize_the_matrix_right_order import MatrixProcessor
from WriteInputfile_Class import WriteInputFile
from EmbeddedStiffnessMatrix_Class import EmbeddedStiffnessMatrix 
from Multiindex_exponents import MultiExponentGenerator
from Duplicate_Input_File import InputFileModifier
from Run_jobs import run_abaqus_job


class TangentStiffnessMatrixHandler:
    """Handles computation and loading of tangent stiffness matrices for different calculation options."""
    computed_keys = {1: set(), 2: set()}  # Key 2 for both options 2 and 3
    
    def __init__(self, option, base_input_filename, non_embedded_nodes, exponent_table, **kwargs):
        """
        Initialize the handler with calculation parameters.
        
        Args:
            option (int): Calculation option (1, 2, or 3)
            base_input_filename (str): Base filename for matrix files
            non_embedded_nodes (list): List of non-embedded node indices
            exponent_table (DataFrame): Table containing exponent information
            **kwargs: Additional arguments specific to each option:
                - Option 1: exponent, part, sign, displacement
                - Option 2: exponent_real, exponent_imag, displacement
                - Option 3: exponent1, exponent2, part1, part2, W_u_dict
        """
        self.option = option
        self.base_input_filename = base_input_filename
        self.non_embedded_nodes = non_embedded_nodes
        self.exponent_table = exponent_table
        
        if option == 1:
            self.exponent = kwargs['exponent']
            self.part = kwargs['part']
            self.sign = kwargs['sign']
            self.displacement = kwargs['displacement']
            self.minimum_distance = kwargs['minimum_distance']
            self.exponent_str = ''.join(map(str, self.exponent))
            row = self.exponent_table[self.exponent_table['exponents'] == self.exponent].iloc[0]
            if self.part == 'real':
                self.k_tg_index = row['K_tg(+Re(w))'] if self.sign == 'pos' else row['K_tg(-Re(w))']
            else:
                self.k_tg_index = row['K_tg(+Im(w))'] if self.sign == 'pos' else row['K_tg(-Im(w))']
            self.mtx_filename = f"{base_input_filename}_{self.k_tg_index}.mtx"
            self.key = self.k_tg_index
            self.cc_exponent_index = row['global_index_cc']
            
        elif option in (2, 3):  # Handle options 2 and 3 together
            if option == 2:
                self.exponent_real = kwargs['exponent_real']
                self.exponent_imag = kwargs['exponent_imag']
                self.displacement = kwargs['displacement']
                self.minimum_distance = kwargs['minimum_distance']
                row_real = self.exponent_table[self.exponent_table['exponents'] == self.exponent_real].iloc[0]
                row_imag = self.exponent_table[self.exponent_table['exponents'] == self.exponent_imag].iloc[0]
                k1 = row_real['K_tg(+Re(w))']
                k2 = row_imag['K_tg(+Im(w))']
            else:  # option 3
                self.exponent1 = kwargs['exponent1']
                self.exponent2 = kwargs['exponent2']
                self.part1 = kwargs['part1']
                self.part2 = kwargs['part2']
                self.minimum_distance = kwargs['minimum_distance']
                self.W_u_dict = kwargs['W_u_dict']
                row1 = self.exponent_table[self.exponent_table['exponents'] == self.exponent1].iloc[0]
                row2 = self.exponent_table[self.exponent_table['exponents'] == self.exponent2].iloc[0]
                k1 = row1['K_tg(+Re(w))'] if self.part1 == 'real' else row1['K_tg(+Im(w))']
                k2 = row2['K_tg(+Re(w))'] if self.part2 == 'real' else row2['K_tg(+Im(w))']
            
            # Normalize key to smallest index first
            a, b = min(k1, k2), max(k1, k2)
            self.key = (a, b)
            self.mtx_filename = f"{base_input_filename}_{a}_{b}.mtx"
            
        else:
            raise ValueError("Invalid option. Choose 1, 2, or 3.")

    def compute_or_load(self):
        """
        Compute the matrix if not already computed, then load it.
        
        Returns:
            The loaded stiffness matrix for the specified configuration
        """
        # Choose the correct set based on option
        key_set = self.computed_keys[1] if self.option == 1 else self.computed_keys[2]
        if self.key not in key_set:
            self._compute()
            key_set.add(self.key)

        return self._load_matrix()

    def _compute(self):
        """Internal method to dispatch computation based on option."""

        if self.option == 1:
            self._compute_option1()
        elif self.option in (2, 3):
            self._compute_option23()



    def _compute_option1(self):
        """Compute all four variants (real/imag × pos/neg) for a single exponent."""
        row = self.exponent_table[self.exponent_table['exponents'] == self.exponent].iloc[0]
        if row['Abaqus']:
            return
            
        cases = [
            ('real', 'pos'),
            ('real', 'neg'),
            ('imag', 'pos'),
            ('imag', 'neg')
        ]
        
        for part, sign in cases:
            calculator = ExponentCalculator_all(
                self.base_input_filename,
                self.exponent,
                self.non_embedded_nodes,
                self.displacement,
                self.exponent_table,
                part,
                sign,
                self.minimum_distance
            )
            calculator.compute()
            
        # Mark exponent and its conjugate as computed
        self.exponent_table.loc[self.exponent_table['exponents'] == self.exponent, 'Abaqus'] = True
        self.exponent_table.loc[self.exponent_table['global_index'] == self.cc_exponent_index, 'Abaqus'] = True

        # Store all four keys in computed set
        indices = [
            row['K_tg(+Re(w))'],
            row['K_tg(-Re(w))'],
            row['K_tg(+Im(w))'],
            row['K_tg(-Im(w))']
        ]
        self.computed_keys[1].update(indices)


    def _compute_option23(self):
        """Compute matrices for options 2 or 3 (real/imag combinations or exponent pairs)."""
        if self.option == 2:
            calculator = ExponentCalculator_single(
                self.base_input_filename,
                self.mtx_filename,
                self.displacement, 
                self.minimum_distance
            )
            calculator.compute()
        else:  # option 3
            cases = [
                ('real', 'real'),
                ('real', 'imag'),
                ('imag', 'real'),
                ('imag', 'imag')
            ]
            
            for part1, part2 in cases:
                row1 = self.exponent_table[self.exponent_table['exponents'] == self.exponent1].iloc[0]
                row2 = self.exponent_table[self.exponent_table['exponents'] == self.exponent2].iloc[0]
                
                k1 = row1['K_tg(+Re(w))'] if part1 == 'real' else row1['K_tg(+Im(w))']
                k2 = row2['K_tg(+Re(w))'] if part2 == 'real' else row2['K_tg(+Im(w))']
                a, b = min(k1, k2), max(k1, k2)
                key = (a, b)
                
                if key in self.computed_keys[2]:
                    continue
                    
                calculator = ExponentCalculatorForPair(
                    self.base_input_filename,
                    self.exponent1,
                    self.exponent2,
                    self.non_embedded_nodes,
                    self.W_u_dict,
                    self.exponent_table,  # Pass exponent_table here
                    part1,
                    part2,
                    self.minimum_distance
                )
                calculator.compute() #1,2
                self.computed_keys[2].add(key)



    def _load_matrix(self):
        """
        Load the matrix from file and extract the non-embedded portion.
        
        Returns:
            The extracted non-embedded stiffness matrix
            
        Raises:
            FileNotFoundError: If the matrix file doesn't exist
        """
        if not os.path.exists(self.mtx_filename):
            raise FileNotFoundError(f"Matrix file not found: {self.mtx_filename}")
        embedded_matrix = EmbeddedStiffnessMatrix(
            self.non_embedded_nodes, 
            self.mtx_filename
        )
        return embedded_matrix.load_and_extract_non_embedded_matrix()



class ExponentCalculator_all:
    """Calculates all tangent stiffness matrix variants (real/imag × positive/negative) for a given exponent."""

    def __init__(self, base_input_filename, exponent, non_embedded_nodes, displacement, exponent_table, part, sign, minimum_distance):
        """
        Initialize the calculator for a specific exponent configuration.
        
        Args:
            base_input_filename (str): Base filename for input/output files
            exponent: The exponent value to compute
            non_embedded_nodes (list): List of non-embedded node indices
            displacement: Complex displacement vector
            exponent_table (DataFrame): Table containing exponent information
            part (str): 'real' or 'imag' part of the displacement to use
            sign (str): 'pos' or 'neg' sign for the displacement
        """
        self.base_input_filename = base_input_filename
        self.exponent = exponent
        self.non_embedded_nodes = non_embedded_nodes
        self.displacement = displacement
        self.exponent_table = exponent_table
        self.part = part
        self.sign = sign
        self.minimum_distance=minimum_distance
        self.exponent_str = ''.join(map(str, exponent))
        
        # Get K_tg index from exponent_table
        row = self.exponent_table[self.exponent_table['exponents'] == self.exponent].iloc[0]
        if self.part == 'real':
            self.k_tg_index = row['K_tg(+Re(w))'] if self.sign == 'pos' else row['K_tg(-Re(w))']
        else:  # 'imag'
            self.k_tg_index = row['K_tg(+Im(w))'] if self.sign == 'pos' else row['K_tg(-Im(w))']
        self.mtx_filename = f"{base_input_filename}_{self.k_tg_index}.mtx"

    def compute(self):
        """
        Perform the full computation process:
        1. Mark exponent as computed
        2. Prepare displacement vector
        3. Create modified input file
        4. Run Abaqus job
        5. Process resulting matrix
        """
        # Set Abaqus flag to True for this exponent
        self.exponent_table.loc[self.exponent_table['exponents'] == self.exponent, 'Abaqus'] = True
        
        # Create displacement vector based on part and sign
        disp_array = np.array(self.displacement, dtype=complex)
        if self.part == 'real':
            disp = disp_array.real
            scale_factor=(0.24*self.minimum_distance/(np.max(np.abs(disp))) if np.any(disp != 0) 
                          else  1)
            print(f"scale factor: {scale_factor}")
            disp= disp*scale_factor
        else:  # 'imag'
            disp = disp_array.imag
            scale_factor=(0.24*self.minimum_distance/(np.max(np.abs(disp))) if np.any(disp != 0) 
                          else  1)
            disp= disp*scale_factor
        if self.sign == 'neg':
            disp = -disp
        print(f"displacement: {disp}")
        # Skip computation if displacement is zero
        if np.all(np.any(disp == 0)):
            print(f"Skipping computation for {self.mtx_filename} - zero displacement")
            return
            
        inp_filename = self.mtx_filename.replace('.mtx', '.inp')
        self._create_input_file(disp, inp_filename)
        self._run_abaqus(self.base_input_filename)
        self._process_matrix(inp_filename)

    def _create_input_file(self, disp, output_filename):
        """
        Create modified input file with the specified displacement.
        
        Args:
            disp: Displacement vector to apply
            output_filename: Name for the modified input file
        """
        flat_disp = disp.flatten().tolist()
        modifier = InputFileModifier(
            input_filename=f"{self.base_input_filename}.inp",
            displacement=flat_disp,
            output_filename=output_filename
        )
        modifier.create_modified_input_files()

    def _run_abaqus(self, inp_filename):
        """
        Execute the Abaqus job with the modified input file.
        
        Args:
            inp_filename: Input file to run through Abaqus
            
        Raises:
            RuntimeError: If the Abaqus job fails
        """
        try:
            run_abaqus_job(inp_filename)
        except Exception as e:
            raise RuntimeError(f"Abaqus job failed for {inp_filename}: {str(e)}")

    def _process_matrix(self, inp_filename):
        """
        Process the resulting matrix file from Abaqus.
        
        Args:
            inp_filename: Base input filename used to locate the matrix file
        """
        mtx_file = os.path.splitext(inp_filename)[0] + '.mtx'
        processor = MatrixProcessor(mtx_file)
        processor.process_matrix_file()




class ExponentCalculator_single:
    """Calculates a single tangent stiffness matrix combining both real and imaginary displacement parts."""

    def __init__(self, base_input_filename, mtx_filename, displacement, minimum_distance):
        """
        Initialize the single matrix calculator.
        
        Args:
            base_input_filename (str): Base filename for input/output files
            mtx_filename (str): Filename for the resulting matrix file
            displacement: Complex displacement vector to use
        """
        self.base_input_filename = base_input_filename
        self.mtx_filename = mtx_filename
        self.displacement = displacement
        self.minimum_distance = minimum_distance

    def compute(self):
        """
        Perform the full computation process:
        1. Combine real and imaginary displacement parts
        2. Create modified input file
        3. Run Abaqus job
        4. Process resulting matrix
        """
        disp_array = np.array(self.displacement, dtype=complex)
        scale_factor_r=(0.24*self.minimum_distance/np.max(np.abs(disp_array.real)) if np.any(disp_array.real != 0) else 1)
        print(f"scale_factor_r {scale_factor_r}") 
        scale_factor_i=(0.24*self.minimum_distance/np.max(np.abs(disp_array.imag)) if np.any(disp_array.imag != 0) else 1) 
        print(f"scale_factor_i {scale_factor_i}") 
        disp_combined_plus = disp_array.real*scale_factor_r +  disp_array.imag*scale_factor_i
        disp_combined_minus = disp_array.real*scale_factor_r -  disp_array.imag*scale_factor_i

        if max(abs(disp_combined_plus)) >= 0.24*self.minimum_distance:
            disp_combined=disp_combined_plus
            sign=1

        else:
            #compute difference of displacements
            disp_combined=disp_combined_minus
            sign=-1


        print(f"++++++++++++++++++++++++++++++CASE 2 SIGN {sign}++++++++++++++++++++++++++++++++++++++++++++") 

        # Skip computation if combined displacement is zero
        if np.all(disp_combined == 0):
            print(f"Skipping computation for {self.mtx_filename} - zero displacement")
            return
            
        inp_filename = self.mtx_filename.replace('.mtx', '.inp')
        self._create_input_file(disp_combined, inp_filename)
        self._run_abaqus(self.base_input_filename)
        self._process_matrix(inp_filename)




    def _create_input_file(self, disp, output_filename):
        """
        Create modified input file with the specified displacement.
        
        Args:
            disp: Displacement vector to apply
            output_filename: Name for the modified input file
        """
        flat_disp = disp.flatten().tolist()
        modifier = InputFileModifier(
            input_filename=f"{self.base_input_filename}.inp",
            displacement=flat_disp,
            output_filename=output_filename
        )
        modifier.create_modified_input_files()

    def _run_abaqus(self, inp_filename):
        """
        Execute the Abaqus job with the modified input file.
        
        Args:
            inp_filename: Input file to run through Abaqus
            
        Raises:
            RuntimeError: If the Abaqus job fails
        """
        try:
            run_abaqus_job(inp_filename)
        except Exception as e:
            raise RuntimeError(f"Abaqus job failed for {inp_filename}: {str(e)}")

    def _process_matrix(self, inp_filename):
        """
        Process the resulting matrix file from Abaqus.
        
        Args:
            inp_filename: Base input filename used to locate the matrix file
        """
        mtx_file = os.path.splitext(inp_filename)[0] + '.mtx'
        processor = MatrixProcessor(mtx_file)
        processor.process_matrix_file()




class ExponentCalculatorForPair:
    """Calculates tangent stiffness matrices for pairs of exponents with specified parts."""

    def __init__(self, base_input_filename, exponent1, exponent2, non_embedded_nodes, W_u_dict, exponent_table, part1, part2, minimum_distance):
        """
        Initialize the pair matrix calculator.
        
        Args:
            base_input_filename (str): Base filename for input/output files
            exponent1: First exponent in the pair
            exponent2: Second exponent in the pair
            non_embedded_nodes (list): List of non-embedded node indices
            W_u_dict (dict): Dictionary mapping exponents to displacement vectors
            exponent_table (DataFrame): Table containing exponent information
            part1 (str): 'real' or 'imag' part for first exponent
            part2 (str): 'real' or 'imag' part for second exponent
        """
        self.base_input_filename = base_input_filename
        self.exponent1 = exponent1
        self.exponent2 = exponent2
        self.non_embedded_nodes = non_embedded_nodes
        self.W_u_dict = W_u_dict
        self.exponent_table = exponent_table  # Store exponent_table
        self.part1 = part1
        self.part2 = part2
        self.minimum_distance=minimum_distance
        
        # Create normalized filename (smallest index first)
        k1 = self._get_k_index(exponent1, part1)
        k2 = self._get_k_index(exponent2, part2)
        a, b = min(k1, k2), max(k1, k2)
        self.mtx_filename = f"{base_input_filename}_{a}_{b}.mtx"
        
    def _get_k_index(self, exponent, part):
        """
        Helper method to get stiffness matrix index from exponent_table.
        
        Args:
            exponent: Exponent to look up
            part: 'real' or 'imag' part specification
            
        Returns:
            The corresponding stiffness matrix index from the table
        """
        # Helper to get stiffness matrix index from exponent_table
        row = self.exponent_table[self.exponent_table['exponents'] == exponent].iloc[0]
        return row['K_tg(+Re(w))'] if part == 'real' else row['K_tg(+Im(w))']

    def compute(self):
        """
        Perform the full computation process for the exponent pair:
        1. Get displacement vectors for both exponents
        2. Combine specified parts
        3. Create modified input file
        4. Run Abaqus job
        5. Process resulting matrix
        """
        disp_vec1 = self.W_u_dict[self.exponent1]
        disp_vec2 = self.W_u_dict[self.exponent2]
        
        # Create displacement vector based on parts
        disp1 = disp_vec1.real if self.part1 == 'real' else disp_vec1.imag
        scale_factor_sigma=(0.24*self.minimum_distance/np.max(np.abs(disp1)) if np.any(disp1 != 0) else 1) 
        print(f"scale_factor_sigma {scale_factor_sigma}")

        disp2 = disp_vec2.real if self.part2 == 'real' else disp_vec2.imag
        scale_factor_nu=(0.24*self.minimum_distance/np.max(np.abs(disp2)) if np.any(disp2 != 0) else 1) 
        print(f"scale_factor_nu {scale_factor_nu}")
        disp_combined_plus = scale_factor_sigma*disp1 + scale_factor_nu*disp2
        disp_combined_minus = scale_factor_sigma*disp1 - scale_factor_nu*disp2


        if max(abs(disp_combined_plus)) >= 0.24*self.minimum_distance:
            disp_combined=disp_combined_plus
            sign=1

        else:
            #compute difference of displacements
            disp_combined=disp_combined_minus
            sign=-1

        print(f"++++++++++++++++++++++++++++++CASE 3 SIGN {sign}++++++++++++++++++++++++++++++++++++++++++++") 
        print(f"disp_combined \n: {disp_combined}")
        # Skip computation if combined displacement is zero
        if np.all(disp_combined == 0):
            print(f"Skipping computation for {self.mtx_filename} - zero displacement")
            return 
            
        inp_filename = self.mtx_filename.replace('.mtx', '.inp')
        self._create_input_file(disp_combined, inp_filename)
        self._run_abaqus(self.base_input_filename)
        self._process_matrix(inp_filename)



    def _create_input_file(self, disp, output_filename):
        """
        Create modified input file with the specified displacement.
        
        Args:
            disp: Displacement vector to apply
            output_filename: Name for the modified input file
        """
        flat_disp = disp.flatten().tolist()
        modifier = InputFileModifier(
            input_filename=f"{self.base_input_filename}.inp",
            displacement=flat_disp,
            output_filename=output_filename
        )
        modifier.create_modified_input_files()

    def _run_abaqus(self, inp_filename):
        """
        Execute the Abaqus job with the modified input file.
        
        Args:
            inp_filename: Input file to run through Abaqus
            
        Raises:
            RuntimeError: If the Abaqus job fails
        """
        try:
            run_abaqus_job(inp_filename)
        except Exception as e:
            raise RuntimeError(f"Abaqus job failed for {inp_filename}: {str(e)}")

    def _process_matrix(self, inp_filename):
        """
        Process the resulting matrix file from Abaqus.
        
        Args:
            inp_filename: Base input filename used to locate the matrix file
        """
        mtx_file = os.path.splitext(inp_filename)[0] + '.mtx'
        processor = MatrixProcessor(mtx_file)
        processor.process_matrix_file()





