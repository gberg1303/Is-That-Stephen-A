import os, csv
import pandas as pd
import numpy as np
from re import search

### Get Filenames and Create Dataframe
for files in os.walk("/Users/jonathangoldberg/Google Drive/Random/Random Fun/Is That Spehen A/Data/Training_Dataset"):
    for filename in files:
        filenames = filename

### Add to the Dataframe
df = pd.DataFrame(filenames, columns=['FileName'])

### Add Darth Vader and Stepehen A Columns DataFrame
df['Darth_Vader'] = df['FileName'].str.contains('Darth_Vader')*1
df['Stephen_A'] = df['FileName'].str.contains('Stephen_A')*1
