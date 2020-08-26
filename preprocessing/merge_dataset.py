# This script merge datasets together

###  Settings  ################################################################
input_paths = ["data/twitter/labelled/full.csv",
               "data/twitter/unlabelled/labelled.csv"]
out_path = "data/twitter/labelled_and_predicted/full.csv"
columns_to_merge = 2 # The X first columns are merged
sep = "\t"
header=None
###############################################################################

import pandas as pd
import os
from pathlib import *

output_dir = os.path.dirname(out_path)
Path(output_dir).mkdir(parents=True, exist_ok=True)

df = pd.DataFrame()
for x in input_paths:
    new_df = pd.read_csv(x, sep=sep, header=header)
    new_df = new_df.iloc[:, :columns_to_merge]
    df = pd.concat([df, new_df])

df.to_csv(out_path, sep=sep, header=header, index=False)
