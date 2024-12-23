import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

class Condinions:
    

    def __init__(self):
        pass
    
    def cond1(self, m, threshold=0.5): 
        prop = QED.qed(m)
        if prop >= threshold:   
            return 1
        else:   
            return 0
        
    def cond2(self, m, threshold_min=1, threshold_max=3): 
        prop = Descriptors.MolLogP(m)
        if prop >= threshold_min and prop <= threshold_max:   
            return 1
        else:   
            return 0
        
    def cond3(self, m, threshold_min=20, threshold_max=140): 
        prop = Descriptors.TPSA(m)
        if prop >= threshold_min and prop <= threshold_max:   
            return 1
        else:   
            return 0
    
    def get_cond(self, smiles_list):
        generated_smiles = [i for i in smiles_list if i != '']
        df = pd.DataFrame(columns=['qed', 'logp', 'tpsa', '0'])
        for smiles in generated_smiles:
            try:
                m = Chem.MolFromSmiles(smiles)
                df.loc[len(df)] = [self.cond1(m), self.cond2(m), self.cond3(m), smiles]
            except:
                continue
        return df
