import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from sympy.physics.wigner import wigner_3j
from sympy import Rational

# Build (J, M) basis list
def build_basis(Jmax):
    return [(J, M) for J in range(Jmax + 1) for M in range(-J, J + 1)]

# Return Hamiltonian matrix element ⟨J,M|H|J',M'⟩
def h_element(J, M, Jp, Mp, B, mu, E):
    if M != Mp:
        return 0.0
    h_rot = B * J * (J + 1) if J == Jp else 0.0
    try:
        w1 = float(wigner_3j(J, 1, Jp, 0, 0, 0).evalf())
        w2 = float(wigner_3j(J, 1, Jp, -M, 0, M).evalf())
        h_dip = -(-1)**M * mu * E * np.sqrt((2*J+1)*(2*Jp+1)) * w1 * w2
    except:
        h_dip = 0.0
    return h_rot + h_dip

# Build sparse Hamiltonian matrix
def build_hamiltonian(basis, B, mu, E):
    dim = len(basis)
    H = lil_matrix((dim, dim))
    for i, (J, M) in enumerate(basis):
        for j, (Jp, Mp) in enumerate(basis):
            val = h_element(J, M, Jp, Mp, B, mu, E)
            if abs(val) > 1e-12:
                H[i, j] = val
    return H.tocsr()

# Plot sparsity pattern with axis labels
def plot_sparsity(H, basis):
    dim = len(basis)
    plt.figure(figsize=(7, 7))
    plt.spy(H, markersize=4, color='blue')
    plt.title("Sparsity Pattern of Hamiltonian Matrix\n$\\langle J, M | H | J', M' \\rangle$", fontsize=14)
    plt.xlabel("Ket index $(J', M')$", fontsize=12)
    plt.ylabel("Bra index $(J, M)$", fontsize=12)
    
    if dim <= 30:
        labels = [f"{J},{M}" for J, M in basis]
        plt.xticks(np.arange(dim), labels, rotation=90, fontsize=6)
        plt.yticks(np.arange(dim), labels, fontsize=6)
    else:
        plt.xticks([]); plt.yticks([])
    
    plt.tight_layout()
    plt.show()

# Diagonalize Hamiltonian: dense for small, sparse for large matrices
def diagonalize(H, num_eig=6):
    dim = H.shape[0]
    if dim <= 300:
        return np.sort(eigh(H.toarray())[0])
    else:
        return np.sort(eigsh(H, k=min(num_eig, dim - 2), which='SA')[0])

# Main execution
def main():
    # Parameters
    Jmax = 10
    B = 20.0
    mu = 1.0
    E = 100.5

    # Build basis and Hamiltonian
    basis = build_basis(Jmax)
    H = build_hamiltonian(basis, B, mu, E)

    # Plot sparsity
    plot_sparsity(H, basis)

    # Diagonalize and show eigenvalues
    eigenvals = diagonalize(H, num_eig=6)
    print("Lowest energy eigenvalues (in units of Bħ²):")
    for i, val in enumerate(eigenvals):
        print(f"  Level {i}: {val:.6f}")

# Run if script
if __name__ == "__main__":
    main()

