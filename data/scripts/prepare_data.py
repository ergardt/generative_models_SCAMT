import pandas as pd
import argparse
from rdkit import Chem

def parameter():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='../database_ChEMBL.csv')
    return parser.parse_args()

def main():
    args = vars(parameter())
    data_path = args['data_path']
    df = pd.read_csv(data_path)
    molecules = list(df.iloc[:100000, 0]) # test code, for full data use: list(df.iloc[:, 0])
    train_df = pd.DataFrame([Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in molecules], columns=[0]) # canonization
    train_df.to_csv('../train_set.csv', index=False)

    with open('../train_set.txt', 'w') as f:
        for mol in molecules:
            f.write(f"{mol}\n")

if __name__ == "__main__":
    main()