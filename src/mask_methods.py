##  
# AUTHOR   : Jackson Taylor
# CREATED  : 3-31-2025
# EDITED   : 3-31-2025
# CONTAINS : Pre-proccessing methods to clean and mask data and output to ./data/clean
##

import os
import pandas as pd

MASK_TOKEN = "<MASK>"
TAB_TOKEN  = "<TAB>"

orig_path  = os.path.abspath(os.path.join("data", "Archive"))
clean_path = os.path.abspath(os.path.join("data", "clean"))

filenames  = ["ft_test.csv", "ft_train.csv", "ft_valid.csv"]

#removes all new line characters
def flatten_methods(df : pd.DataFrame, method_col = "cleaned_method"):
    for i in range(len(df)):
        df.loc[i, method_col] = df[method_col][i].replace("\n", "")
        df.loc[i, method_col] = df[method_col][i].replace("\r", "")
    return df

#replaces all tabs with token versions
def tokenize_tabs(df : pd.DataFrame, method_col = "cleaned_method"):
    for i in range(len(df)):
        df.loc[i, method_col] = df[method_col][i].replace("    ", " " + TAB_TOKEN + " ")
    return df

#splits the string along the subsqeuent tokens, tokens, contained in the string. Throws error if tokens not contained
def split_on_tokens(string : str, tokens : list):
    start_ind  = 0
    while True:
        all_tokens_found = True
        if string.find(tokens[0], start_ind) < 0:
            raise IndexError("tokens not found for string: " + string + ", tokens: " + " ".join(tokens))
        left_string  = string[:string.find(tokens[0], start_ind)]
        right_string = string[string.find(tokens[0], start_ind) + len(tokens[0]):].lstrip()
        for token in tokens[1:]:
            if not right_string.find(token):
                right_string = right_string[right_string.find(token) + len(token):].lstrip()
            else:
                all_tokens_found = False
                break
        if all_tokens_found:
            return (left_string, right_string)
        else:
            start_ind = string.find(tokens[0], start_ind) + 1
        
#masks all if statements in the dataframe df and replaces them with the mask token "<MASK>"
def mask_if_stmts(df : pd.DataFrame, method_col = "cleaned_method", if_col = "target_block"):
    for i in range(len(df)):
        if_stmt : list = df[if_col][i].split(" ")
        method  : str  = df[method_col][i]
        split_method = split_on_tokens(method, if_stmt)
        df.loc[i, method_col] = split_method[0].rstrip() + " " + MASK_TOKEN + " " + split_method[1].lstrip()
    return df

for name in filenames:
    input_path  = os.path.join(orig_path, name)
    output_path = os.path.join(clean_path, name)

    df = pd.read_csv(input_path)

    df = mask_if_stmts(df)
    df = flatten_methods(df)
    df = tokenize_tabs(df)

    df.to_csv(output_path, header=True, index=False)

