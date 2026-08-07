import pandas as pd
import pickle

class ResultPrinter:
    def __init__(self, dpim):
        """Initialize result printer with DPIM system data.
        
        Args:
            dpim: Data structure containing computation results
        """
        self.dpim = dpim  # Reference to main DPIM data container

    def _print_reduced_dynamics(self, n_total):
        """Print the reduced dynamics coefficients (f) in a readable format.
        
        Args:
            n_total: Number of terms to display (initial subset)
        """
        print("\nf (reduced dynamics):")
        # Create row labels for modes
        index = [f"Mode {i}" for i in range(self.dpim.n_modes)]
        # Create DataFrame with appropriate headers
        df = pd.DataFrame(self.dpim.f[:, :n_total], index=index, columns=self.dpim.column_headers[:n_total])
        
        # Print full table if small, otherwise chunked
        if n_total <= 7:
            print(df)
        else:
            self._print_chunked_dataframe(df, n_total)

    def _print_chunked_dataframe(self, df, n_total):
        """Print large DataFrames in manageable chunks.
        
        Args:
            df: DataFrame to print
            n_total: Total number of columns
        """
        # Determine number of chunks based on size
        n_chunks = 2 if n_total <= 14 else 3
        chunk_size = (n_total + n_chunks - 1) // n_chunks
        start = 0
        # Print each chunk
        for _ in range(n_chunks):
            end = min(start + chunk_size, n_total)
            print(df.iloc[:, start:end])
            start = end

    def _create_result_dataframes(self):
        """Create DataFrames for all result matrices (Wu, Wv, f).
        
        Returns:
            Tuple of DataFrames for Wu, Wv, and f
        """
        # Create DataFrames with polynomial term headers
        df_wu = pd.DataFrame(self.dpim.Wu, columns=self.dpim.column_headers)
        df_wv = pd.DataFrame(self.dpim.Wv, columns=self.dpim.column_headers)
        
        # Debug print if needed
        #print(f"f without changed: {self.dpim.f}")
        #print(f"Imprime expoentes:\n {self.dpim.column_headers}")
        
        # Create f DataFrame with mode-based indexing
        df_f = pd.DataFrame(self.dpim.f, index=[f"Mode {i}" for i in range(self.dpim.n_modes)], 
                          columns=self.dpim.column_headers)
        return df_wu, df_wv, df_f

    def _print_chunked_results(self, title, df):
        """Print a result matrix in column chunks.
        
        Args:
            title: Section header
            df: DataFrame to print
        """
        print(title)
        chunk_size = 5  # Columns per chunk
        # Iterate through column ranges
        for start in range(0, len(self.dpim.column_headers), chunk_size):
            end = min(start + chunk_size, len(self.dpim.column_headers))
            print(f"Columns {start} to {end-1}:")
            print(df.iloc[:, start:end])

    def _print_resonance_relations(self):
        """Print detected resonance relationships between terms and modes."""
        print("\nResonance relations:")
        # Iterate through all mode-resonance term pairs
        for mode, terms in self.dpim.resonance_relations.items():
            print(f"Mode {mode} resonant with terms: {terms}")

    def _print_final_results(self, df_wu, df_wv, df_f):
        """Print all final results in organized sections.
        
        Args:
            df_wu: DataFrame of Wu coefficients
            df_wv: DataFrame of Wv coefficients
            df_f: DataFrame of reduced dynamics
        """
        print("\nFinal Results:")
        # Print each matrix in chunks
        self._print_chunked_results("Wu:", df_wu)
        self._print_chunked_results("Wv:", df_wv)
        self._print_chunked_results("f:", df_f)
        # Print resonance information
        self._print_resonance_relations()

    def _save_results_to_file(self):
        """Save reduced dynamics coefficients (f) to a pickle file."""
        with open('my_array.pkl','wb') as f:
            pickle.dump(self.dpim.f, f)

    def save_to_txt(self, filename): # folde1r_name/folder2/file_name
        """Save f (coefficients) and column_headers (multi-indices) to a text file in the specified format."""
        with open(filename, 'w') as file:
            # Format A_coeffs (complex coefficients) with one element per line
            file.write("A_coeffs = np.array([\n")
            for i in range(self.dpim.f.shape[0]):
                file.write("    [\n")
                # Format each complex number with scientific notation
                for j in range(self.dpim.f.shape[1]):
                    c = self.dpim.f[i, j]
                    real = c.real
                    imag = c.imag
                    sign = '+' if imag >= 0 else '-'
                    element = f"        {real:.8e}{sign}{abs(imag):.8e}j"
                    # Add comma except for last element
                    if j < self.dpim.f.shape[1] - 1:
                        file.write(element + ",\n")
                    else:
                        file.write(element + "\n")
                # Close inner array
                if i < self.dpim.f.shape[0] - 1:
                    file.write("    ],\n")
                else:
                    file.write("    ]\n")
            file.write("])\n\n")

            # Format multi_indices_single_element (exponents)
            file.write("multi_indices_single_element = np.array([\n")
            for idx, exp_str in enumerate(self.dpim.column_headers):
                # Parse string representation of tuple
                clean_str = exp_str.strip('()')
                numbers = clean_str.split(',')
                int_list = [int(num.strip()) for num in numbers]
                # Format as [1,0,0,0] without spaces
                inner_array = "[" + ",".join(str(x) for x in int_list) + "]"
                
                # Add comma except for last element
                if idx < len(self.dpim.column_headers) - 1:
                    file.write(f"    {inner_array},\n")
                else:
                    file.write(f"    {inner_array}\n")
            file.write("])\n")


    def save_Wu_to_file(self, filename):
        """Save Wu matrix and multi-indices to a text file."""
        with open(filename, 'w') as file:
            # Write Wu matrix
            file.write("Wu_matrix = np.array([\n")
            for i in range(self.dpim.Wu.shape[0]):
                file.write("    [\n")
                for j in range(self.dpim.Wu.shape[1]):
                    c = self.dpim.Wu[i, j]
                    real = c.real
                    imag = c.imag
                    sign = '+' if imag >= 0 else '-'
                    element = f"        {real:.8e}{sign}{abs(imag):.8e}j"
                    if j < self.dpim.Wu.shape[1] - 1:
                        file.write(element + ",\n")
                    else:
                        file.write(element + "\n")
                if i < self.dpim.Wu.shape[0] - 1:
                    file.write("    ],\n")
                else:
                    file.write("    ]\n")
            file.write("])\n\n")

            # Write multi-indices
            file.write("multi_indices_single_element = np.array([\n")
            for idx, exp_str in enumerate(self.dpim.column_headers):
                clean_str = exp_str.strip('()')
                numbers = clean_str.split(',')
                int_list = [int(num.strip()) for num in numbers]
                inner_array = "[" + ",".join(str(x) for x in int_list) + "]"
                if idx < len(self.dpim.column_headers) - 1:
                    file.write(f"    {inner_array},\n")
                else:
                    file.write(f"    {inner_array}\n")
            file.write("])\n")

    def save_Wv_to_file(self, filename):
        """Save Wv matrix and multi-indices to a text file."""
        with open(filename, 'w') as file:
            # Write Wv matrix
            file.write("Wv_matrix = np.array([\n")
            for i in range(self.dpim.Wv.shape[0]):
                file.write("    [\n")
                for j in range(self.dpim.Wv.shape[1]):
                    c = self.dpim.Wv[i, j]
                    real = c.real
                    imag = c.imag
                    sign = '+' if imag >= 0 else '-'
                    element = f"        {real:.8e}{sign}{abs(imag):.8e}j"
                    if j < self.dpim.Wv.shape[1] - 1:
                        file.write(element + ",\n")
                    else:
                        file.write(element + "\n")
                if i < self.dpim.Wv.shape[0] - 1:
                    file.write("    ],\n")
                else:
                    file.write("    ]\n")
            file.write("])\n\n")

            # Write multi-indices (same as for Wu)
            file.write("multi_indices_single_element = np.array([\n")
            for idx, exp_str in enumerate(self.dpim.column_headers):
                clean_str = exp_str.strip('()')
                numbers = clean_str.split(',')
                int_list = [int(num.strip()) for num in numbers]
                inner_array = "[" + ",".join(str(x) for x in int_list) + "]"
                if idx < len(self.dpim.column_headers) - 1:
                    file.write(f"    {inner_array},\n")
                else:
                    file.write(f"    {inner_array}\n")
            file.write("])\n")
 