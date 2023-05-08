import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ete3

import os
from tqdm import tqdm
import pickle

from src.phylogenetic_signal import PagelsLambda
from src.ihmp import get_diffs
from src.greengenes import (
    parse_similarity_map,
    merge_otu_table,
    combine_similarity_maps,
)

# I load OTU tables using the basic iHMP scripts from earlier notebooks

data = {
    # name: get_diffs(name, get_abundances=True, log=False)  # undo log transform
    name: get_diffs(name, get_abundances=False, log=False)
    for name in ["ibd", "moms", "t2d"]
}

# Skip all sanity checks

# Get lambdas for each cutoff
from multiprocessing import Pool

MEMOIZED = True
PRECISION = 2
similarity_map99 = parse_similarity_map(99)

for name in ["ibd", "moms", "t2d"]:
    lambdas = []

    # It's actually faster to go by site, since inversion is O(n^2):
    for site in data[name].index.get_level_values("site").unique():
        data_site = data[name].loc[site]
        data_site = data_site.loc[:, data_site.sum(axis=0) > 0]

        similarity_map = similarity_map99  # Restart at 99
        for x in [99, 97, 94, 91, 88, 85, 82, 79, 76, 73, 70, 67, 64, 61]:

            # Non-redundancy check
            site_str = site.replace(" ", "_")
            outpath = (
                f"./results/growth_cluster/{name}_{site_str}_pls_cutoff{x}.tsv"
            )
            pklpath = (
                f"./results/growth_cluster/{name}_{site_str}_pls_cutoff{x}.pkl"
            )
            if os.path.exists(outpath):
                print(f"Skipping {outpath} because it already exists.")
                lambdas_cutoff_df = pd.read_csv(outpath, sep="\t")

            else:
                lambdas_cutoff = []

                # Get GreenGenes data
                if x == 99:  # Special case
                    similarity_map = similarity_map99
                else:
                    similarity_map = combine_similarity_maps(
                        [similarity_map, parse_similarity_map(x)]
                    )

                # Filter OTU table
                clustered = merge_otu_table(data_site, similarity_map)
                s = clustered.sum(axis=1)
                print(
                    x,
                    clustered.shape,
                    f"{s.min():.3f} {s.mean():.3f} {s.max():.3f}",
                )

                # Filter tree, init PagelsLambda
                tree = ete3.Tree(
                    f"./greengenes/data/gg_13_5_otus/trees/{x}_otus.tree",
                    format=1,
                    quoted_node_names=True,
                )
                tree.prune(clustered.columns)
                pl = PagelsLambda(tree, memoized=MEMOIZED)
                print(f"Tree has {len(pl.tree)} leaves.")

                # # Change everything to float16 - this will save a lot of memory
                # clustered = clustered.astype(np.float16)
                # pl.C = pl.C.astype(np.float16)

                # Get lambdas
                for sample in tqdm(clustered.index):
                    pl.fit(
                        clustered.loc[sample],
                        precision=PRECISION,
                        method="optimize",
                    )
                    lambdas_cutoff.append(
                        {
                            "sample": sample,
                            "lambda": np.round(pl.lam, PRECISION),
                        }
                    )

                lambdas_cutoff_df = pd.DataFrame(lambdas_cutoff)
                lambdas_cutoff_df.to_csv(outpath, sep="\t")

            lambdas_cutoff_df["cutoff"] = x
            lambdas_cutoff_df["dataset"] = name
            lambdas_cutoff_df["site"] = site
            lambdas.append(lambdas_cutoff_df)

    lambdas_df = pd.concat(lambdas)
