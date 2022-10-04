import pandas as pd

import numpy as np


df = pd.DataFrame(np.random.randn(6, 4), columns=list('ABCD'))

df['A'].loc[0] = np.nan

s = df.to_numpy()
k = np.isnan(s).any() 
print(k)