
import numpy as np
from scipy.sparse.linalg import eigsh
from EmbeddedStiffnessMatrix_Class import EmbeddedStiffnessMatrix
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg import lobpcg

def newton_raphson_eigen_dense(K, M, x0, lambda0, tol=1e-10, max_iter=10, epsilon=1e-12):
    """
    Newton-Raphson solver for generalized eigenvalue problem M x omega^2 = K x.
    Uses dense matrices and np.linalg.solve for linear systems.
    
    Parameters:
        K, M: Dense matrices from the eigenproblem.
        x0: Initial eigenvector approximation.
        lambda0: Initial eigenvalue approximation (omega^2).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
        epsilon: Regularization parameter for near-singular systems.
        
    Returns:
        x: Improved eigenvector.
        lambda_: Improved eigenvalue.
    """
    # Ensure matrices are in dense format
    if hasattr(K, 'toarray'):
        K = K.toarray()
    if hasattr(M, 'toarray'):
        M = M.toarray()
    
    n = K.shape[0]
    
    # Normalize initial vector
    x = x0 / np.sqrt(x0 @ M @ x0)
    lambda_ = lambda0
    
    for iter in range(max_iter):
        # Compute residuals
        res = K @ x - lambda_ * M @ x
        norm_err = 0.5 * (x @ M @ x - 1)
        
        # Check convergence
        res_norm = np.linalg.norm(res)
        if res_norm < tol * np.linalg.norm(K @ x) and abs(norm_err) < tol:
            print(f"Converged in {iter} iterations.")
            break
        
        # Form the regularized matrix A = (K - lambda*M) + epsilon*I
        A_reg = (K - lambda_ * M) + epsilon * np.eye(n)
        
        # Right-hand sides
        b1 = -res  # - (K - lambda*M)*x
        J12 = -M @ x  # -M*x
        
        # Solve A_reg * y1 = b1
        y1 = np.linalg.solve(A_reg, b1)
        
        # Solve A_reg * y2 = J12
        y2 = np.linalg.solve(A_reg, J12)
        
        # Compute dlambda
        numerator = x @ M @ y1 - (-norm_err)  # x.T*M*y1 - b2
        denominator = x @ M @ y2
        dlambda = numerator / denominator
        
        # Compute dx
        dx = y1 - y2 * dlambda
        
        # Update eigenpair
        x += dx
        lambda_ += dlambda
        
        # Normalize eigenvector
        x /= np.sqrt(x @ M @ x)
        
        print(f"Iteration {iter+1}: |res| = {res_norm:.2e}, |norm_err| = {abs(norm_err):.2e}")
    else:
        print("Maximum iterations reached without convergence.")
    
    return x, lambda_

def eigensolutions_conservative(K, M, k_mode, sigma, refine=True):
    """
    Solves the conservative (undamped) eigenvalue problem K·Φ = M·Φ·ω²
    
    Args:
        K (np.ndarray): Stiffness matrix (n x n)
        M (np.ndarray): Mass matrix (n x n)
        refine (bool): Whether to apply Newton-Raphson refinement
        
    Returns:
        tuple: (frequencies, eigenvectors) where:
            frequencies (np.ndarray): Natural angular frequencies (rad/s) (n,)
            eigenvectors (np.ndarray): Mode shapes (n x n)
    """
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigsh(
        A=K, k=k_mode, M=M, sigma=sigma, which='LA', maxiter=100000
    )
    """print(f"eigenvalues: {eigenvectors[:,0]}")
    eigenvalues, eigenvectors=lobpcg(A=K, X=eigenvectors, B=K, maxiter=8000, largest=False)
    print(f"eigenvalues: {eigenvectors[:,0]}")"""
    # Refine eigenvalues and eigenvectors using Newton-Raphson
    """if refine:
        refined_eigenvectors = np.zeros_like(eigenvectors)
        refined_eigenvalues = np.zeros_like(eigenvalues)
        for i in range(k_mode):
            x0 = eigenvectors[:, i]
            lambda0 = eigenvalues[i]
            x_refined, lambda_refined = newton_raphson_eigen_dense(K, M, x0, lambda0)
            refined_eigenvectors[:, i] = x_refined
            refined_eigenvalues[i] = lambda_refined
        eigenvectors = refined_eigenvectors
        eigenvalues = refined_eigenvalues"""
    
    # Verify eigenvalues are real and positive
    assert np.all(np.isreal(eigenvalues)), "Eigenvalues should be real"
    assert np.all(eigenvalues > 0), "Eigenvalues should be positive"
    
    # Convert to angular frequencies (ω = √λ)
    frequencies = np.sqrt(eigenvalues)
    
    return frequencies, eigenvectors

def real_part_Rayleigh_damp(a, b, frequencies):
    """[Unchanged]"""
    return -0.5 * (a + b * frequencies**2)

def eigenvalues_similar_damp(frequencies, real_part=None):
    """[Unchanged]"""
    if real_part is None:
        return np.concatenate((1j * frequencies, -1j * frequencies))
    
    discriminant = real_part**2 - frequencies**2
    sqrt_disc = np.lib.scimath.sqrt(discriminant)
    evals_plus = real_part + sqrt_disc
    evals_minus = real_part - sqrt_disc
    return np.concatenate((evals_plus, evals_minus))

def eigenvectors_normalization(eigenvectors, M, k_mode):
    """[Unchanged]"""
    normalized_right = np.zeros_like(eigenvectors, dtype=complex)
    for i in range(eigenvectors.shape[1]):
        phi = eigenvectors[:, i]
        normalization_factor = np.sqrt(phi.T @ M @ phi)
        normalized_right[:, i] = phi / normalization_factor

    repeated_right = np.zeros((normalized_right.shape[0], k_mode*2), dtype=complex)
    for i in range(k_mode):
        repeated_right[:, i] = normalized_right[:, i]
        repeated_right[:, i + k_mode] = normalized_right[:, i]

    return repeated_right

def eigensolutions_damped(K, M, sigma, a, b, k):
    """
    Solves proportionally damped eigenvalue problem 
    (Mλ² + Cλ + K)Φ = 0 with C = aM + bK
    
    Args:
        K (np.ndarray): Stiffness matrix (n x n)
        M (np.ndarray): Mass matrix (n x n)
        a (float): Mass damping coefficient
        b (float): Stiffness damping coefficient
        
    Returns:
        tuple: (eigenvectors_normalized, eigenvalues, eigenvectors_left)
    """
    k_mode = k // 2
    freqs, eigenvectors = eigensolutions_conservative(K, M, k_mode, sigma, refine=True)
    
    real_parts = real_part_Rayleigh_damp(a, b, freqs)
    eigenvalues = eigenvalues_similar_damp(freqs, real_parts)
    
    eigenvectors_normalized = eigenvectors_normalization(eigenvectors, M, k_mode)
    eigenvectors_left = eigenvectors_normalized.copy()
    
    return eigenvectors_normalized, eigenvalues, eigenvectors_left



if __name__ == "__main__":
    # (Your existing main code remains unchanged)
    #non_embedded_nodes = [1, 2, 3, 4, 401, 402, 403, 404, 405, 406, 407, 408, 805, 806, 807, 808, 809, 810, 811, 812, 1209, 1210, 1211, 1212]  # List of nodes that are not embedded
    #non_embedded_nodes = [1,2,3,4]  # List of nodes that are not embedded
    non_embedded_nodes = [1,2,3,4,5,6,7,8]
    mass_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'Meu_input_file_mass_matrix_E_210e9.mtx.mtx')
    #mass_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'COMSOL_mass_matrix.mtx')
    #mass_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'strange_element_mass_matrix.mtx')
    M=mass_matrix.load_and_extract_non_embedded_matrix()
    stiffness_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'Two_elements_stiffness_matrix.mtx')
    #stiffness_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'COMSOL_stiffness_matrix.mtx')
    #stiffness_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'strange_element_stiffness_matrix.mtx')
    K=stiffness_matrix.load_and_extract_non_embedded_matrix()
    a, b = 0.0, 0.0
    C = a * M + b * K
    sigma = 0.1

    Phi, lambda_, left_eigenvectors, eigenfrequencies = eigensolutions_damped(
        K=K, 
        M=M,  
        sigma=sigma,
        a=a, 
        b=b, 
        k=6
    )
    #non_embedded_nodes = [4,6,7,8]
    #mass_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'COMSOL_mass_matrix.mtx')
    #mass_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'strange_element_mass_matrix.mtx')
    #M=mass_matrix.load_and_extract_non_embedded_matrix()
    #stiffness_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'Meu_input_file.mtx')
    #stiffness_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'COMSOL_stiffness_matrix.mtx')
    #stiffness_matrix=EmbeddedStiffnessMatrix(non_embedded_nodes, 'strange_element_stiffness_matrix.mtx')
    #K=stiffness_matrix.load_and_extract_non_embedded_matrix()
    #Phi_c, lambda_c, left_eigenvectors_c, eigenfrequencies_c = eigensolutions_damped(
    #    K=K, 
    #    M=M,  
    #    sigma=sigma,
    #    a=a, 
    #    b=b, 
    #    k=2
    #)
    print(f"phi:\n{Phi}")
    #print(f"phi:\n{Phi_c}")
    print(f"eigenvalues:\n{lambda_}")
    print(f"eigenfrequency:\n{eigenfrequencies}")
    #print(f"erro compare phi:\n {(Phi-Phi_c)/np.linalg.norm(Phi)}")
    

    LHS=K@Phi[:,:5]

    print(f"LHS:{LHS}")

    RHS=M@Phi[:,:5]*(eigenfrequencies**2)

    print(f"RHS:{RHS}")

    print(f"erro: {np.linalg.norm(np.abs(LHS-RHS))/(np.linalg.norm(np.abs(LHS)))}")