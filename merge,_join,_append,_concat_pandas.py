# -*- coding: utf-8 -*-
"""Merge, Join, Append, Concat - Pandas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IzYc5OMarXNN93V8yG6Qu3MarZnw-0tj
"""

# !pip install tensorflow

# Pandas Combining DataFrames

import numpy as np
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
tips.head()

# Merge

tips_bill = tips.groupby(['sex', 'smoker'])[['total_bill', 'tip']].sum()
tips_tip = tips.groupby(['sex', 'smoker'])[['total_bill', 'tip']].sum()

del tips_bill['tip']
del tips_tip['total_bill']

tips_bill

tips_tip

pd.merge?

# we can merge on the indexes
pd.merge(tips_bill, tips_tip, right_index = True, left_index = True)

# we can reset the indexes and then merge on the columns - perhabs the easiest way

pd.merge(tips_bill.reset_index(),
         tips_tip.reset_index(), on = ['sex', 'smoker'])

# it. an actually infer the above - but be very careful with this

pd.merge(
    tips_bill.reset_index(),
    tips_tip.reset_index()
)

# it can merge on partial column and index

pd.merge(
    tips_bill.reset_index(),
    tips_tip,
    left_on = ['sex', 'smoker'],
    right_index = True
)

# it can do interesting combination
tips_bill_strange = tips_bill.reset_index(level = 0)
tips_bill_strange

pd.merge(
    tips_tip.reset_index(),
    tips_bill_strange,
    on = ['sex', 'smoker']
)

#we can do any SQL like functionality

pd.merge(
    tips_bill.reset_index(),
    tips_tip.reset_index().head(2),
    how = 'left'
)

pd.merge(
    tips_bill.reset_index(),
    tips_tip.reset_index().head(2),
    how = 'inner'
)

pd.merge(
    tips_bill.reset_index(),
    tips_tip.reset_index().head(2),
    how = 'right'
)

# and if you add an indicator

pd.merge(
    tips_bill.reset_index().head(3),
    tips_tip.reset_index().tail(3),
    how = 'outer',
    indicator = True
)

# it can handle column with same name
pd.merge(
    tips_bill.reset_index(), 
    tips_bill.reset_index(),
    left_index = True,
    right_index = True,
    suffixes = ['_left', '_right']
)

# it can handle column with same name
pd.merge(
    tips_bill, 
    tips_bill,
    left_index = True,
    right_index = True,
    suffixes = ['_left', '_right']
)

"""##Concatenation"""

tips_bill

tips_tip

# row wise joining
pd.concat([tips_bill, tips_bill, tips_tip], sort = False)

# this will does column wise
pd.concat([tips_bill, tips_tip], axis = 1)

"""Note that axis = 1 specifies the axis along which the concatenation is performed. If axis = 0, the concatenation would be performed vertically, i.e., row-wise."""

# and finally this will add on the dataset where it's from
pd.concat([tips_bill, tips_tip], sort = False, keys = ['num0', 'num2'])