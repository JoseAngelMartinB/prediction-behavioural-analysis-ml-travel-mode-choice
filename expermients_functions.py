"""
Common functions for experiments executed in the paper.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas
"""

import numpy as np
import random
from collections import Counter, defaultdict
import re
from sklearn.utils import shuffle
import pandas as pd
import sys

## Definitions for AMPCA and GMPCA
def AMPCA(proba, y):
    sum = 0
    i = 0
    for sel_mode in y:
        sum = sum + proba[i,sel_mode]
        i += 1
    N = i-1
    return sum/N

def CEL(proba, y):
    sum = 0
    i = 0
    for sel_mode in y:
        sum = sum + np.log(proba[i,sel_mode])
        i += 1
    N = i-1
    return -sum/N

def GMPCA(proba, y):
    return np.exp(-CEL(proba, y))


# Sample a dataset grouped by `groups` and stratified by `y`
# Source: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


## Export latex tables
def pandas_to_latex(df_table, latex_file, vertical_bars=False, right_align_first_column=True, align_rest='c', header=True, index=False,
                    escape=False, multicolumn=False, midrule_jump=[], fit_width=False, caption="", label="", **kwargs) -> None:
    """
    Function that augments pandas DataFrame.to_latex() capability.
    :param df_table: dataframe
    :param latex_file: filename to write latex table code to
    :param vertical_bars: Add vertical bars to the table (note that latex's booktabs table format that pandas uses is
                          incompatible with vertical bars, so the top/mid/bottom rules are changed to hlines.
    :param right_align_first_column: Allows option to turn off right-aligned first column
    :param align_rest: Alignment for the rest of columns
    :param header: Whether or not to display the header
    :param index: Whether or not to display the index labels
    :param escape: Whether or not to escape latex commands. Set to false to pass deliberate latex commands yourself
    :param multicolumn: Enable better handling for multi-index column headers - adds midrules
    :param midrule_jump: List of indices containing the column headers you want to avoid adding midrules
    :param fit_width: Fit table size to text width
    :param caption: Latex caption
    :param label: Latex lable
    :param kwargs: additional arguments to pass through to DataFrame.to_latex()
    :return: None
    """
    n = len(df_table.columns) + int(index)

    if right_align_first_column:
        cols = 'r' + align_rest * (n - 1)
    else:
        cols = align_rest * n

    if vertical_bars:
        # Add the vertical lines
        cols = '|' + '|'.join(cols) + '|'

    latex = df_table.to_latex(escape=escape, index=index, column_format=cols, header=header, multicolumn=multicolumn, caption=caption, label=label,
                              **kwargs)

    if vertical_bars:
        # Remove the booktabs rules since they are incompatible with vertical lines
        latex = re.sub(r'\\(top|mid|bottom)rule', r'\\hline', latex)

    # Multicolumn improvements - center level 1 headers and add midrules
    if multicolumn:
        latex = latex.replace(r'{l}', r'{c}')

        offset = int(index)
        midrule_str = ''
        for i, col in enumerate(df_table.columns.levels[0]):
            indices = np.nonzero(np.array(df_table.columns.codes[0]) == i)[0]
            hstart = 1 + offset + indices[0]
            hend = 1 + offset + indices[-1]
            if i not in midrule_jump:
                midrule_str += rf'\cmidrule(lr){{{hstart}-{hend}}} '

        latex_lines = latex.splitlines()
        latex_lines.insert(7, midrule_str)

        if fit_width == True:
            latex_lines.insert(4, '\\resizebox{\\textwidth}{!}{') # fit_width
            latex_lines.insert(-1, '}') # End of fit_width

        latex = '\n'.join(latex_lines)

    with open(latex_file, 'w') as f:
        f.write(latex)


## Interquartile range
def IQR(data, k=1.5):
    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)
    iqr = Q3 - Q1
    return data[(data >= Q1-k*iqr) & (data <= Q3+k*iqr)]


# Balance dataset
def balance(X, y, n_obs, n_alt):
    # X and y are put together in the same dataframe to perform the same transformations on both
    unbalanced_data = X.copy()
    unbalanced_data['y'] = y

    # Create a dataframe to store the balanced dataset
    balanced_data = pd.DataFrame()

    # The average number of observations per alternative on the training set is computed
    n = int(np.ceil(n_obs/n_alt))

    for alt in range(0, n_alt):
        # Split the data associated with each alternative
        data_alt = unbalanced_data.loc[unbalanced_data['y'] == alt]
        alt_obs = data_alt.shape[0]

        if alt_obs < n:
            # Upsampling. We apply a random sample with repetition
            X_new = data_alt.sample(n=n-alt_obs, replace=True)
            balanced_data = balanced_data.append(data_alt)
            balanced_data = balanced_data.append(X_new)
            print("Alternativa: %s. Aplicado Upsampling.  Antes: %d --- Ahora: %d" % (alt, alt_obs, alt_obs + X_new.shape[0]))
            sys.stdout.flush()

        elif alt_obs >= n:
            # Downsampling. We apply a random sample of n elements without repetition
            X_new = data_alt.sample(n=n, replace=False)
            balanced_data = balanced_data.append(X_new)
            print("Alternativa: %s. Aplicado Downsampling.  Antes: %d --- Ahora: %d" % (alt, alt_obs, X_new.shape[0]))
            sys.stdout.flush()

    balanced_data = shuffle(balanced_data)
    balanced_data = balanced_data.reset_index(drop=True)

    # Split data into features (X) and label (y)
    balanced_X = balanced_data.loc[:, balanced_data.columns != "y"]
    balanced_y = balanced_data["y"]
    
    return (balanced_X, balanced_y)