"""Smiles to Image pre processor pipeline."""

from pathlib import Path
from typing import Any, List

import joblib
from joblib import cpu_count, delayed, Parallel
from kedro.pipeline import node, Pipeline
import numpy as np
import pandas as pd
import pyarrow.feather as feather
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
    in_path: Path,
    out_path: Path,
    transformer_path: Path = transformer_path,
    df_chunks: int = 10000,
) -> None:
    """Convert Smiles to Images.

    Args:
        in_path (Path): Input CSV path
        out_path (Path): Output CSV path
        transformer_path (Path): Path of transformer. Defaults to transformer_path.
        df_chunks (int): df chunks to create. Defaults to 100.
    """
    df_full = pd.read_csv(project_root_path.joinpath(in_path).absolute())
    df_len = len(df_full)
    splits = [x for x in range(0, df_len + 1, (df_len + 1) // df_chunks)]
    split_tuples = [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]
    dfs = [df_full.iloc[start:end] for start, end in split_tuples]
    grid_transformer = joblib.load(transformer_path)
    Parallel(n_jobs=cpu_count())(
        delayed(chunked_df_to_feather)(df, i, grid_transformer, in_path, split_tuples)
        for i, df in enumerate(tqdm(dfs))
    )


def chunked_df_to_feather(
    df: pd.DataFrame, i: int, grid_transformer: Any, in_path: Path, split_tuples: tuple
) -> None:
    """Save chunked dfs to feather files.

    Args:
        df (pd.DataFrame): pandas Dataframe
        i (int): dataframe position
        grid_transformer (Any): grid transformer
        in_path (Path): Path
        split_tuples (tuple): Split ranges
    """
    smiles = df["smiles"]
    R = []
    for chunk in tqdm(chunker(smiles, 100)):
        mols = [Chem.MolFromSmiles(smi) for smi in chunk]
        e = ecfp_counts(mols)
        R += [e]
    ecfp = np.concatenate(R)
    molecule_imgs = grid_transformer.transform(ecfp)
    imgs_df = pd.DataFrame(molecule_imgs.reshape(-1, 1024))
    global_index_df = pd.DataFrame(
        [
            x
            for x in range(
                split_tuples[i][0],
                split_tuples[i][1],
            )
        ],
        columns=["global_index"],
    )
    global_index_df["target"] = df.iloc[:, 0].to_list()
    global_index_df[[x for x in range(0, 1024)]] = imgs_df
    feather.write_feather(
        global_index_df,
        project_root_path.joinpath(in_path)
        .parents[0]
        .joinpath(f"sm_to_imgs_{i}.feather.lz4")
        .absolute(),
        compression="lz4",
    )


convert_smiles_to_imgs_node = node(
    func=convert_smiles_to_imgs,
    inputs=["params:in_path", "params:out_path"],
    outputs=None,
)

# Assemble nodes into a pipeline
convert_smiles_to_imgs_pipeline = Pipeline([convert_smiles_to_imgs_node])
