import os, csv
import pandas as pd
import numpy as np
from re import search

### Get Filenames and Create Dataframe
dirname = os.path.dirname(__file__)
for files in os.walk(os.path.join(dirname, "Data/Training_Dataset")):
    for filename in files:
        filenames = filename

### Add to the Dataframe
df = pd.DataFrame(filenames, columns=['FileName'])

### Add Darth Vader and Stepehen A Columns DataFrame
df['Other'] = 1 - df['FileName'].str.contains('Stephen_A')*1
df['Stephen_A'] = df['FileName'].str.contains('Stephen_A')*1