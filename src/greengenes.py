import numpy as np
import pandas as pd
import ete3
from tqdm import tqdm
from Bio.SeqIO import parse
from collections import defaultdict


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

    # Get rep IDs as well
    repset = set(
        [x.id for x in parse(f"{gg_dir}/rep_set/{cutoff}_otus.fasta", "fasta")]
    )

    otu2rep = {}
    rep2otus = {}
    leaves = set()
    with open(f"{gg_dir}/otus/{cutoff}_otu_map.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            # First element is just the index
            otu_ids = line.strip().split("\t")[1:]
            reps = set(otu_ids) & repset
            for otu_id in otu_ids:
                otu2rep[otu_id] = reps
            for rep in reps:
                leaves.add(rep)
                if rep not in rep2otus:
                    rep2otus[rep] = set(otu_ids)
                else:
                    rep2otus[rep] = rep2otus[rep] | set(otu_ids)

            # for otu_id in otu_ids:
            #     otu_keys[otu_id] = None
            # for otu_id in otu_ids:
            #     if otu_id in leafset:
            #         otu_keys[otu_id] = otu_ids
            #         new_leaves.append(otu_id)
            #         # break

            # Which one is the rep?
    return otu2rep, rep2otus, tree


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
    original: pd.DataFrame, otu2rep: dict, rep2otus: dict
) -> pd.DataFrame:
    """Aggregate rows of an OTU table."""

    # # Aggregate rows
    # # out_df = pd.DataFrame(index=otu_table.index)
    # keys = {
    #     k: v
    #     for k, v in otu_keys.items()
    #     if v is not None and np.any([x in otu_table.columns for x in v])
    # }
    # out = []
    # for otu, vals in tqdm(keys.items()):
    #     vals_in_table = [val for val in vals if val in otu_table.columns]
    #     # out_df[otu] = np.sum(otu_table.loc[vals_in_table, :], axis=1)
    #     sum = np.sum(otu_table[vals_in_table], axis=1)
    #     sum.name = otu
    #     out.append(sum)

    # # Use concat for speed
    # out_df = pd.concat(out, axis=1)

    # # Drop columns with pseudocounts only, or NaNs
    # out_df = out_df.loc[:, (out_df > 1e-10).any(axis=0)]
    # out_df = out_df.loc[:, out_df.notnull().any(axis=0)]

    # return out_df
    out_df = pd.DataFrame(
        index=original.index,
        columns=[],  # [x for x in rep2otus if x in original.columns],
        data=0,
    )
    # otu2rep = {k: v for k, v in otu2rep.items() if k in original.columns}
    original_otu_set = set(original.columns)

    # for otu, reps in otu2rep.items():
    for otu in original.columns:
        try:
            reps = otu2rep[otu]
        except KeyError:
            reps = set()

        if len(reps) == 0:
            # print("No reps for OTU", otu)
            # out_df.loc[:, otu] = original[otu]  # Go straight through
            continue

        elif len(reps) == 1:
            rep = reps.pop()
            # if rep not in out_df.columns:
            #     out_df.columns.append(rep)
            if rep not in out_df.columns:
                out_df.loc[:, rep] = 0
            out_df.loc[:, rep] += original[otu]

        else:  # len(reps) > 1

            # Try restricting to original table OTUs
            new_reps = original_otu_set & reps
            if len(new_reps) == 0:
                rep = reps.pop()
            else:
                rep = new_reps.pop()

            if rep not in out_df.columns:
                out_df.loc[:, rep] = 0
            out_df.loc[:, rep] += original[otu]

    return out_df


def parse_similarity_map(n: int, otus_dir="./greengenes/data/gg_13_5_otus"):
    """Map cluster --> OTUs"""
    map_path = f"{otus_dir}/otus/{int(n)}_otu_map.txt"
    map_dict = defaultdict(list)
    map_str = open(map_path).read()
    for line in map_str.strip().split("\n"):
        _, cluster_id, *otu_ids = line.split("\t")
        map_dict[cluster_id].extend([cluster_id, *otu_ids])
    return map_dict


def combine_similarity_maps(map_dicts):
    """Map cluster --> OTUs iteratively"""

    # Our first map maps 99% OTUs to 97% OTUs
    # We want to go 99 --> 97 --> 95 --> 94 --> etc...
    # We want to map cluster --> OTUs still

    higher_level_map = map_dicts[0]
    for map_dict in map_dicts[1:]:
        # Generate mapping 99 --> x
        new_map_dict = defaultdict(list)
        for cluster_id, otus in map_dict.items():
            for otu in otus:
                new_map_dict[cluster_id].extend(higher_level_map[otu])
            # for otu in otus:
            #     new_map_dict[cluster_id].extend(higher_level_map[cluster_id])
            #     # Principle: if a later cluster exists, it existed in all
            #     # earlier maps
        higher_level_map = new_map_dict

    return higher_level_map


def merge_otu_table(otu_table, map_dict, drop=True):
    new_otu_table = pd.DataFrame(index=otu_table.index)
    for cluster_id, otus in map_dict.items():
        existing_otus = [otu for otu in otus if otu in otu_table.columns]
        if existing_otus:
            new_otu_table[cluster_id] = otu_table[existing_otus].sum(axis=1)
    if drop:
        new_otu_table = new_otu_table.loc[:, new_otu_table.sum(axis=0) > 0]
    return new_otu_table


def calculate_fb_ratio(
    sample: pd.Series,
    f_path="./greengenes/data/firmicutes.txt",
    b_path="./greengenes/data/bacteroidetes.txt",
) -> float:
    """Given a greengenes-annotated series, calculate F/B ratio"""
    # Load annotations
    f_ids = set(open(f_path).read().strip().split("\n"))
    b_ids = set(open(b_path).read().strip().split("\n"))

    f_total = sample[sample.index.isin(f_ids)].sum()
    b_total = sample[sample.index.isin(b_ids)].sum()

    if b_total == 0:
        return np.nan
    else:
        return f_total / b_total
