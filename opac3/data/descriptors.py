# opac3/data/descriptors.py

from rdkit import Chem
from rdkit.Chem import Descriptors
from ase import Atoms
import numpy as np
from scipy.linalg import eigh
import openbabel
from openbabel import openbabel
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

def compute_rdkit_descriptors(mol: Chem.Mol) -> dict:
    """
    Computes RDKit descriptors for the given molecule.
    """
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        # Add more descriptors as needed
    }
    return descriptors

def compute_coulomb_matrix_eigenvalues(atoms: Atoms) -> np.ndarray:
    """
    Computes the eigenvalues of the Coulomb matrix for the given ASE Atoms object.
    Returns the eigenvalues sorted in descending order.
    """
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    n_atoms = len(atoms)
    coulomb_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                coulomb_matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
            else:
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance > 1e-8:
                    coulomb_matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j]) / distance
                else:
                    coulomb_matrix[i, j] = 0.0  # Avoid division by zero
    # Compute eigenvalues
    eigenvalues = eigh(coulomb_matrix, eigvals_only=True)
    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues

def compute_descriptors(atoms: Atoms) -> dict:
    """
    Computes descriptors for the given molecule represented as an ASE Atoms object.
    """
    # Convert ASE Atoms to RDKit Mol using Open Babel
    mol = atoms_to_rdkit_mol(atoms)
    if mol is None:
        raise ValueError("Could not convert Atoms to RDKit Mol.")
    
    # Compute RDKit descriptors
    descriptors = compute_rdkit_descriptors(mol)
    
    # Compute Coulomb matrix eigenvalues
    eigenvalues = compute_coulomb_matrix_eigenvalues(atoms)
    # Use a fixed length for eigenvalues (e.g., pad or truncate to length 20)
    max_eigenvalues = 20
    eigenvalues_padded = np.zeros(max_eigenvalues)
    n_eig = min(len(eigenvalues), max_eigenvalues)
    eigenvalues_padded[:n_eig] = eigenvalues[:n_eig]
    for i in range(max_eigenvalues):
        descriptors[f'CoulombEig_{i}'] = eigenvalues_padded[i]
    return descriptors

def atoms_to_rdkit_mol(atoms: Atoms) -> Chem.Mol:
    """
    Converts an ASE Atoms object to an RDKit Mol object using Open Babel for bond perception.
    """
    from rdkit import Chem
    import openbabel as ob
    from openbabel import openbabel
    from io import StringIO
    import sys

    # Convert ASE Atoms to XYZ string
    xyz_str = StringIO()
    atoms.write(xyz_str, format='xyz')
    xyz_content = xyz_str.getvalue()
    xyz_str.close()

    # Initialize Open Babel conversion
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("xyz")
    obConversion.SetOutFormat("mol")

    # Read molecule from XYZ string
    obMol = openbabel.OBMol()
    obConversion.ReadString(obMol, xyz_content)

    # Perceive bonds and assign bond orders
    obMol.ConnectTheDots()
    obMol.PerceiveBondOrders()

    # Convert OBMol to Mol block
    mol_block = obConversion.WriteString(obMol)

    # Create RDKit Mol from Mol block
    mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
    if mol is None:
        logger.warning("RDKit failed to create Mol from Mol block.")
        return None

    return mol
