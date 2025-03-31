##  
# AUTHOR   : Jackson Taylor
# CREATED  : 3-31-2025
# EDITED   : 3-31-2025
# CONTAINS : Pre-proccessing methods to clean data and output cleaned methods to output file
##

import os
import pandas as pd

orig_path  = os.path.abspath(os.path.join("..", "data", "Archive"))
clean_path = os.path.abspath(os.path.join("..", "data", "clean"))

filenames  = ["ft_text.csv", "ft_train.csv", "ft_valid.csv"]

for name in filenames:
    input_path  = os.path.join(orig_path, name)
    output_paht = os.path.join(clean_path, name) 

