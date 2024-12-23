import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(import_path)+'/../../conditions')

import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')

from calc_cond import Condinions

def main():
    classifier = Condinions()
    train_df = pd.read_csv('../train_set.csv')

    train_df_cond = pd.DataFrame(columns=['qed', 'logp', 'tpsa'])
    for smiles in tqdm(train_df['0']):
        m = Chem.MolFromSmiles(smiles)
        train_df_cond.loc[len(train_df_cond)] = [classifier.cond1(m), classifier.cond2(m), classifier.cond3(m)]
    train_df_cond.to_csv('../train_cond.csv', index=False)

if __name__ == "__main__":
    main()