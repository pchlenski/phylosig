import numpy as np
import pandas as pd
import ete3
from tqdm import tqdm


def generate_otu_keys(
    cutoff: float,
    gg_dir: str = "greengenes/data/gg_13_5_otus",
) -> (dict, list, ete3.Tree):
    """Return a mapping from OTU ID to aggregate ID"""
    # First, load reference tree
    tree = ete3.Tree(
        f"{gg_dir}/trees/{cutoff}_otus.tree", format=1, quoted_node_names=True
    )
    leafset = set(tree.get_leaf_names())

    otu_keys = {}
    new_leaves = []
    with open(f"{gg_dir}/otus/{cutoff}_otu_map.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            otu_ids = line.strip().split("\t")
            # otu_keys[otu_ids[-1]] = otu_ids
            # for otu_id in otu_ids[:-1]:
            #     otu_keys[otu_id] = None
            # # Map first OTU -> all its neighbors
            # new_leaves.append(otu_ids[-1])

            # # We need to find which leaf is actually in the tree
            # # Check that there is exactly

            # Same as the commented bit above, but we find the index of the
            # leaf in the tree instead of using the last element
            for otu_id in otu_ids:
                otu_keys[otu_id] = None
            for otu_id in otu_ids:
                if otu_id in leafset:
                    otu_keys[otu_id] = otu_ids
                    new_leaves.append(otu_id)
                    # break
    return otu_keys, new_leaves, tree


# def make_cov_df(cov: np.ndarray, leaves: list) -> pd.DataFrame:
#     """Convert a covariance matrix to a dataframe"""
#     return pd.DataFrame(cov, index=leaves, columns=leaves)


def aggregate_tree(tree: ete3.Tree, leaves: dict) -> ete3.Tree:
    """Aggregate leaves of a tree"""
    new_tree = tree.copy()
    new_tree.prune(leaves)
    return new_tree


# def aggregate_covariance_matrix(
#     covs: pd.DataFrame, otu_keys: dict
# ) -> np.ndarray:
#     """Aggregate rows and columns of a covariance matrix."""

#     n = len(leaves)
#     assert covs.shape == (n, n)

#     # Aggregate rows and columns
#     out_matrix = pd.DataFrame()
#     for otu, vals in otu_keys.items():
#         if vals is None:
#             continue  # Skip OTUs which are not the first in their group
#         else:
#             # idx = leaves.index(otu)
#             # to_average = [leaves.index(val) for val in vals]
#             # Aggregate rows and columns
#             out_matrix.loc[otu, :] = np.mean(covs[vals, :], axis=0)
#             out_matrix.loc[:, otu] = np.mean(covs[:, vals], axis=1)

#         # Intentionally not handling KeyError here, because we want to know
#         # if there are any leaves that are not mapped.

#     return out_matrix


def aggregate_otu_table(
    otu_table: pd.DataFrame, otu_keys: dict
) -> pd.DataFrame:
    """Aggregate rows of an OTU table."""

    # Aggregate rows
    # out_df = pd.DataFrame(index=otu_table.index)
    out = []
    for otu, vals in tqdm(otu_keys.items()):
        if vals is None:
            continue  # Skip OTUs which are not the first in their group
        else:
            vals_in_table = [val for val in vals if val in otu_table.index]
            # out_df[otu] = np.sum(otu_table.loc[vals_in_table, :], axis=1)
            sum = np.sum(otu_table.loc[vals_in_table, :], axis=0)
            sum.name = otu
            out.append(sum)

    # Use concat for speed
    out_df = pd.concat(out, axis=1).T

    # Drop columns with all zeros or NaNs
    out_df = out_df.loc[:, (out_df != 0).any(axis=0)]
    out_df = out_df.loc[:, out_df.notnull().any(axis=0)]

    return out_df
