import ete3
import numpy as np
import pandas as pd

from tqdm import tqdm

from .phylogenetic_signal import PagelsLambda


def strip_df(df):
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    return df


def get_diffs(
    dataset, top_n=None, get_abundances=False, include_metadata=False
):
    """Loads all our data!"""
    if dataset == "ibd":
        prefix = "/home/phil/DATA/ihmp/IBD"
    elif dataset == "moms":
        prefix = "/home/phil/DATA/ihmp/MOMS-PI"
    elif dataset == "t2d":
        prefix = "/home/phil/DATA/ihmp/T2D"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    otu_path = f"{prefix}/16s_otus.csv"
    metadata_path = f"{prefix}/16s_metadata.tsv"
    manifest_path = f"{prefix}/16s_manifest.tsv"

    # Load data
    otus = strip_df(pd.read_csv(otu_path, dtype={0: str}))
    otus = otus.set_index(otus.columns[0]).T
    metadata = strip_df(pd.read_table(metadata_path, skipinitialspace=True))
    manifest = strip_df(pd.read_table(manifest_path, skipinitialspace=True))

    # OTU index naming + sample filtering will depend on dataset
    if dataset == "ibd":
        otus.index = [x.split("_")[0] for x in otus.index]  # Works for IBD
        manifest = manifest[
            manifest["urls"].str.endswith("P_taxonomy_closed_reference.biom")
        ]
        manifest["sample"] = [
            x.split("/")[-1].split(".")[0].split("_")[0]
            for x in manifest["urls"]
        ]
    elif dataset in ["moms", "t2d"]:
        manifest["sample"] = [x.split("/")[-1] for x in manifest["urls"]]

    # Add filename to metadata
    manifest = manifest[["sample_id", "sample"]]
    metadata_merged = pd.merge(metadata, manifest, on="sample_id", how="inner")

    # Simplify metadata
    metadata_premerge = metadata_merged.copy()
    metadata_merged = metadata_merged[
        ["subject_id", "sample_body_site", "sample", "visit_number"]
    ]
    metadata_merged.columns = ["patient", "site", "sample", "visit"]
    metadata_merged = metadata_merged.drop_duplicates()

    # Metadata processing and merge with OTU table
    otus_merged = pd.merge(
        otus, metadata_merged, left_index=True, right_on="sample", how="inner"
    )
    otus_merged = otus_merged.drop("sample", axis=1)
    otus_merged = otus_merged.set_index(["site", "patient", "visit"])

    # Normalize OTUs
    otus_merged /= otus_merged.sum(axis=1).values[:, None]

    # Filter to top N OTUs
    if top_n is not None:
        top_otus = (
            otus_merged.sum(axis=0).sort_values(ascending=False).index[:top_n]
        )
        otus_merged = otus_merged[top_otus]

    # Take log
    otus_merged = np.log(otus_merged + 1e-10)  # need pseudocount to avoid -inf

    # Compute diffs at patient level:
    if get_abundances:
        if include_metadata:
            return otus_merged, metadata_premerge
        else:
            return otus_merged

    else:
        diffs = otus_merged.groupby(level=[0, 1]).diff().dropna()
        assert len(diffs) == len(otus_merged) - len(
            otus_merged.groupby(level=[0, 1])
        )
        assert len(diffs) < len(otus)  # Avoid merge issues

    # return otus, manifest, metadata, metadata_merged, otus_merged, diffs
    if include_metadata:
        return diffs, metadata_premerge
    else:
        return diffs


def prune_tree(df, tree_path):
    # Get tree
    tree = ete3.Tree(tree_path, format=1, quoted_node_names=True)

    # How much overlap is there?
    overlap = len(set(tree.get_leaf_names()) & set(df.columns)) / len(
        set(df.columns)
    )
    if overlap < 1:
        print(f"WARNING: {overlap*100:.2f}% overlap between tree and dataframe")

    # Get tree and dataframe in agreement
    tree.prune(list(df.columns))  # Prune tree to OTUs in dataset
    df.reindex(
        columns=tree.get_leaf_names()
    )  # Reindex dataframe to OTUs in tree

    pl = PagelsLambda(tree)

    return pl, tree, df


def pagels_dataframe(df, pl):  # , keys=None):
    # # Get tree
    # tree = ete3.Tree(tree_path, format=1, quoted_node_names=True)

    # # How much overlap is there?
    # overlap = len(set(tree.get_leaf_names()) & set(df.columns)) / len(
    #     set(df.columns)
    # )
    # if overlap < 1:
    #     print(f"WARNING: {overlap*100:.2f}% overlap between tree and dataframe")

    # # Get tree and dataframe in agreement
    # tree.prune(list(df.columns))  # Prune tree to OTUs in dataset
    # df.reindex(
    #     columns=tree.get_leaf_names()
    # )  # Reindex dataframe to OTUs in tree

    pls = pd.Series(index=df.index, name="lambda", dtype=float)
    for i in tqdm(range(len(df))):
        try:
            row = df.iloc[i]
            x = row.values.reshape(-1, 1)  # TODO: support DFs in PagelsLambda
            pl.fit(x)
            # pls[row.name[0]].append(pl.lam) # Separate by disease
            pls.iloc[i] = pl.lam
        except:
            pls.iloc[i] = np.nan

    return pls, tree
