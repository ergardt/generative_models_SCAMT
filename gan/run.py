import torch
import pickle as pi
import os
import yaml
import pandas as pd
import ast
import argparse
from scripts.model import MolGen
from scripts.tokenizer import Tokenizer


def generate(num_molecules, model_path, molecules_path):
    gan_mol = pi.load(open(model_path, 'rb'))
    smiles_list = gan_mol.generate_n(num_molecules)
    df = pd.DataFrame(smiles_list, columns=["SMILES"])
    df.to_csv(molecules_path, index=False)

def train(data_path,
           model_path, 
           bs, 
           lr_optim, 
           lr_discr, 
           log_path,
        #    label_smoothing,
        #    label_smoothing_params,
           num_gen_iterations,
        #    reward_clamp,
        #    update_baseline_weights,
        #    entropy_weight,
           gen_clip_grad_value,
        #    add_validity,
           epochs, 
           device):
    # with open(data_path) as f:
    #     data = f.read().strip().split('\n')

    # print("=== Tokenizer Debug ===")
    # tokenizer = Tokenizer(data)
    # print("Vocab size:", tokenizer.vocab_size)
    # print("Mapping:", tokenizer.mapping)
    # print("Inv mapping:", tokenizer.inv_mapping)

    # # Пример токенизации
    # sample_smiles = data[0]
    # encoded = tokenizer.encode_smile(sample_smiles)  # это torch.Tensor
    # print(f"Encoded:  {encoded}")

    # # Декодируем: преобразуем тензоры в int
    # decoded = ''.join(tokenizer.inv_mapping[int(i)] for i in encoded[:-1])  # без <eos>
    # print(f"Original: {sample_smiles}")
    # print(f"Decoded:  {decoded}")
    # assert sample_smiles == decoded, "Tokenizer mismatch!"

    print('load data...')
    data = []
    with open(data_path, "r") as f:
        for line in f.readlines()[1:]:
            data.append(line.split("\n")[0])
    # data = data[:100000]/
    print(len(data))
    print(f"Первые 3 SMILES: {data[:3]}")

    print('training model...')
    gan_mol = MolGen(data, 
                    hidden_dim=128, 
                    lr_optim=lr_optim, 
                    lr_discr=lr_discr, 
                    log_path =  log_path,
                    # label_smoothing = label_smoothing,
                    # label_smoothing_params = label_smoothing_params,
                    num_gen_iterations = num_gen_iterations,
                    # reward_clamp = reward_clamp,
                    # update_baseline_weights = update_baseline_weights,
                    # entropy_weight=entropy_weight,
                    gen_clip_grad_value=gen_clip_grad_value,
                    # add_validity=add_validity,
                    model_path=model_path,
                    device=device)
    loader = gan_mol.create_dataloader(data, batch_size=bs, shuffle=True, num_workers=1)
    gan_mol.train_n_steps(loader, max_epoch=epochs, evaluate_every=50)
    pi.dump(gan_mol, open(model_path, 'wb'))

def parameter():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='generation', choices=['generation', 'training'])
    # parser.add_argument('--data_path', type=str, default='../data/database_ChEMBL.csv')
    # parser.add_argument('--model_path', type=str, default='checkpoints/gan_mol.pkl')
    # parser.add_argument('--molecules_path', type=str, default='results/generated_molecules_gan.csv')
    # parser.add_argument('--bs', type=int, default=512)
    # parser.add_argument('--lr_optim', type=float, default=1e-3)
    # parser.add_argument('--lr_discr', type=float, default=1e-3)
    # parser.add_argument('--epochs', type=int, default=1)
    # parser.add_argument('--num_molecules', type=int, default=1000)

    # args = vars(parameter())
    args, _ = parser.parse_known_args()

    config_path = args.config
    print(f"Loading config from: {os.path.abspath(config_path)}")  # 🔍 Debug line

    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        print("Loaded config keys:", list(config.keys()))  # 🔍 Debug line
    else:
        print(f"Config file NOT FOUND: {config_path}")

    tuple_params = [
        'label_smoothing_params',
        'update_baseline_weights'
    ]

    for param in tuple_params:
        if param in config:
            value = config[param]
            if isinstance(value, str):
                try:
                    # Безопасно преобразуем строку в Python-объект: "(0, 0.1)" → (0, 0.1)
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, (list, tuple)):
                        config[param] = tuple(parsed)  # или list, как вам удобнее
                    else:
                        raise ValueError(f"Expected list/tuple, got {type(parsed)}")
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Failed to parse {param} = {value!r}: {e}")
            elif isinstance(value, (list, tuple)):
                config[param] = tuple(value)  # нормализуем в кортеж
            else:
                raise TypeError(f"{param} must be a list, tuple, or string representation thereof. Got: {type(value)}")

    config['mode'] = args.mode
    return argparse.Namespace(**config)

def main():
    args = vars(parameter())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = args['mode']
    data_path = args['data_path']
    model_path = args['model_path']
    molecules_path = args['molecules_path']
    bs = args['bs']
    lr_optim = args['lr_optim']
    lr_discr = args['lr_discr']
    epochs = args['epochs']
    num_molecules = args['num_molecules']
    log_path = args['log_path']
    # label_smoothing = args['label_smoothing']
    # label_smoothing_params = args['label_smoothing_params']
    num_gen_iterations = args['num_gen_iterations']
    # reward_clamp = args['reward_clamp']
    # update_baseline_weights = args['update_baseline_weights']
    # entropy_weight = args['entropy_weight']
    gen_clip_grad_value = args['gen_clip_grad_value']
    # add_validity=args['add_validity']

    if mode == 'training':
        print(f"Запуск обучения на устройстве: {device}")
        train(
            data_path,
            model_path,
            bs, 
            lr_optim,
            lr_discr, 
            log_path = log_path,
            # label_smoothing = label_smoothing,
            # label_smoothing_params = label_smoothing_params,
            num_gen_iterations = num_gen_iterations,
            # reward_clamp = reward_clamp,
            # update_baseline_weights=  update_baseline_weights,
            # entropy_weight=entropy_weight,
            gen_clip_grad_value=gen_clip_grad_value,
            # add_validity=add_validity,
            epochs=epochs, 
            device=device)
    
    if mode == 'generation':
        generate(num_molecules, model_path, molecules_path)

if __name__ == "__main__":
    main()