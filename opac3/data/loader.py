# opac3/data/loader.py

import os
from ase import Atoms
from ase.io import read
from typing import List
import glob
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

def read_xyz_files(directory: str) -> List[Atoms]:
    """
    Reads all XYZ files in the given directory and returns a list of ASE Atoms objects.
    Supports multi-molecule XYZ files.
    """
    xyz_files = glob.glob(os.path.join(directory, "*.xyz"))
    molecules = []
    for file in xyz_files:
        # Read all molecules in the file
        mols_in_file = read(file, index=':')
        if isinstance(mols_in_file, Atoms):
            molecules.append(mols_in_file)
        else:
            molecules.extend(mols_in_file)
    logger.info(f"Read {len(molecules)} molecules from {directory}.")
    return molecules
