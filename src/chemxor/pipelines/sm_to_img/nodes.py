"Smiles to Images nodes."

from pathlib import Path
from typing import List

import joblib
from kedro.pipeline import node
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from chemxor.utils import get_project_root_path


project_root_path = get_project_root_path()
default_path = project_root_path.joinpath(
    "data/01_raw/ModelPreds/eos2r5a/ersilia_output.csv"
)
transformer_path = project_root_path.joinpath("data/06_models/grid_transformer.joblib")


def chunker(seq: List, size: int) -> List:
    """Create bacthes for processing smiles.

    Args:
        seq (List): Array of smiles
        size (int): batch size

    Returns:
        List: List of batches
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def ecfp_counts(mols: List) -> List:
    """Create ECFPs from batch of smiles.

    Args:
        mols (List): batch of smiles

    Returns:
        List: batch of ECFPs
    """
    fps = [
        AllChem.GetMorganFingerprint(mol, radius=3, useCounts=True, useFeatures=True)
        for mol in mols
    ]
    nfp = np.zeros((len(fps), 1024), np.uint8)
    for i, fp in enumerate(fps):
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % 1024
            nfp[i, nidx] += int(v)
    return nfp


def convert_smiles_to_imgs(
    in_path: Path, out_path: Path, transformer_path: Path = transformer_path
) -> None:
    """Convert Smiles to Images.

    Args:
        in_path (Path): Input CSV path
        out_path (Path): Output CSV path
        transformer_path (Path): Path of transformer. Defaults to transformer_path.
    """
    df_full = pd.read_csv(in_path.absolute())
    try:
        molecule_imgs_df = pd.read_csv(out_path.absolute())
    except Exception:
        molecule_imgs_df = pd.DataFrame()

    smiles = df_full[0]
    R = []
    for chunk in tqdm(chunker(smiles, 10000)):
        mols = [Chem.MolFromSmiles(smi) for smi in chunk]
        e = ecfp_counts(mols)
        R += [e]
    ecfp = np.concatenate(R)

    grid_transformer = joblib.load(transformer_path)

    molecule_imgs = grid_transformer.transform(ecfp)
    molecule_imgs_df.append(molecule_imgs).to_csv(out_path, index=False)


convert_smiles_to_imgs_node = node(
    func=convert_smiles_to_imgs,
    inputs=["parmas:in_path", "params:out_path"],
    outputs=None,
)