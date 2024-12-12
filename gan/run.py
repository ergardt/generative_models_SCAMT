import torch
import pickle as pi
import pandas as pd
import argparse
from scripts.model import MolGen

def generate(num_molecules, model_path, molecules_path):
    gan_mol = pi.load(open(model_path, 'rb'))
    smiles_list = gan_mol.generate_n(num_molecules)
    df = pd.DataFrame(smiles_list, columns=["SMILES"])
    df.to_csv(molecules_path, index=False)

def train(data_path, model_path, bs, lr, epochs, device):
    print('load data...')
    data = []
    with open(data_path, "r") as f:
        for line in f.readlines()[1:]:
            data.append(line.split("\n")[0])
    print('training model...')
    gan_mol = MolGen(data, hidden_dim=128, lr=lr, device=device)
    loader = gan_mol.create_dataloader(data, batch_size=bs, shuffle=True, num_workers=1)
    gan_mol.train_n_steps(loader, max_epoch=epochs, evaluate_every=50)
    pi.dump(gan_mol, open(model_path, 'wb'))

def parameter():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--function', type=str, default='generation', choices=['generation', 'training'])
    parser.add_argument('--data_path', type=str, default='../data/database_ChEMBL.csv')
    parser.add_argument('--model_path', type=str, default='checkpoints/gan_mol.pkl')
    parser.add_argument('--molecules_path', type=str, default='results/generated_molecules_gan.csv')
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_molecules', type=int, default=1000)
    return parser.parse_args()

if __name__ == "__main__":
    args = vars(parameter())
    device = 'cuda' if torch.cuda.is_available() is False else 'cpu'
    function = args['function']
    data_path = args['data_path']
    model_path = args['model_path']
    molecules_path = args['molecules_path']
    bs = args['bs']
    lr = args['lr']
    epochs = args['epochs']
    num_molecules = args['num_molecules']

    if function == 'training':
        train(data_path, model_path, bs, lr, epochs, device)
    
    if function == 'generation':
        generate(num_molecules, model_path, molecules_path)
