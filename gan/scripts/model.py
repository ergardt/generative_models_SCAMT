import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch import nn
import torch.utils.data as torch_data
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pickle as pi

from scripts.layers import Generator, RecurrentDiscriminator
from scripts.tokenizer import Tokenizer


class MolGen(nn.Module):

    def __init__(self, 
                 data, 
                 hidden_dim=128, 
                 lr_optim=1e-3, 
                 lr_discr=1e-3, 
                 log_path =  './molgen_logs/tmp',
                 label_smoothing = False,
                 label_smoothing_params = [0, 1],
                 num_gen_iterations = 1,
                 reward_clamp = False,
                 update_baseline_weights = [0.9, 0.1],
                 entropy_weight = 0.01,
                 gen_clip_grad_value=0.1,
                 add_validity=False,
                 model_path='/mnt/tank/scratch/aergardt/generative_models/gan/checkpoints/tmp.pkl',
                 device='cpu'
                 ):
        """[summary]

        Args:
            data (list[str]): [description]
            hidden_dim (int, optional): [description]. Defaults to 128.
            lr ([type], optional): learning rate. Defaults to 1e-3.
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.
        """
        super().__init__()

        self.device = device

        self.hidden_dim = hidden_dim

        self.tokenizer = Tokenizer(data)

        self.generator = Generator(
            latent_dim=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            start_token=self.tokenizer.start_token,  # no need token
            end_token=self.tokenizer.end_token,
        ).to(device)

        self.discriminator = RecurrentDiscriminator(
            hidden_size=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            start_token=self.tokenizer.start_token,
            bidirectional=True
        ).to(device)

        self.generator_optim = torch.optim.Adam(
            self.generator.parameters(), lr=lr_optim)
        
        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_discr)

        self.b = 0.  # baseline reward
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        self.history = {'step': [],
                        'loss_disc': [], 
                        'loss_gen': [], 
                        'mean_reward': [],
                        'mean_valid_100': []}
        self.global_step = 0
        self.num_gen_iterations =  num_gen_iterations
        self.label_smoothing = label_smoothing
        self.label_smoothing_params = label_smoothing_params
        print("DEBUG: label_smoothing_params =", self.label_smoothing_params)
        print("DEBUG: type =", type(self.label_smoothing_params))
        print("DEBUG: len =", len(self.label_smoothing_params) if hasattr(self.label_smoothing_params, '__len__') else 'N/A')
        self.reward_clamp = reward_clamp
        self.update_baseline_weights = update_baseline_weights
        self.entropy_weight = entropy_weight
        self.gen_clip_grad_value = gen_clip_grad_value
        self.add_validity=add_validity
        self.model_path = model_path

        print("=== Generator Debug ===")
        print("Generator vocab_size:", self.generator.vocab_size)
        print("Tokenizer vocab_size:", self.tokenizer.vocab_size)
        print("Start token (gen):", self.generator.start_token)
        print("Start token (tok):", self.tokenizer.start_token)


    def sample_latent(self, batch_size):
        """Sample from latent space

        Args:
            batch_size (int): number of samples

        Returns:
            torch.Tensor: [batch_size, self.hidden_dim]
        """
        return torch.randn(batch_size, self.hidden_dim).to(self.device)

    def discriminator_loss(self, x, y, is_real):
        """Discriminator loss

        Args:
            x (torch.LongTensor): input sequence [batch_size, max_len]
            y (torch.LongTensor): sequence label (zeros from generatoe, ones from real data)
                                  [batch_size, max_len]

        Returns:
            loss value
        """

        y_pred, mask = self.discriminator(x).values()

        # loss = F.binary_cross_entropy(
        #     y_pred, y.float(), reduction='none') * mask
        # loss = loss.sum() / mask.sum()

        # y_pred_masked = y_pred * mask
        # loss = (y_pred_masked * y.unsqueeze(-1)).sum() / mask.sum()

        masked_output = y_pred * mask
        mean_output = masked_output.sum() / mask.sum()

        # return loss
        if is_real:
        # Хотим, чтобы D(real) был МАКСИМАЛЬНЫМ → лосс = -D(real)
            return -mean_output
        else:
            # Хотим, чтобы D(fake) был МИНИМАЛЬНЫМ → лосс = +D(fake)
            return mean_output

    def train_step(self, x):
        """One training step with built-in logging and plotting."""
        # print("\n=== TRAIN STEP DEBUG ===")
        # print("Real batch shape:", x.shape)
        # print("Real SMILES example:", self.get_mapped(x[0].cpu().numpy()))
        
        # z = self.sample_latent(2)  # маленький батч
        # gen_out = self.generator(z, max_len=20)  # короткие SMILES
        # x_gen = gen_out['x']
        
        # # Декодируем ДО подачи в дискриминатор
        # print("\nGenerated token tensors (raw):")
        # print(x_gen[0])
        
        # # Заменяем padding (-1 → 0) для дискриминатора
        # # x_gen_for_disc = torch.where(x_gen == -1, torch.zeros_like(x_gen), x_gen)
        
        # # Декодируем для человека
        # lengths = (x_gen > 0).sum(1)
        # smiles_list = []
        # for i in range(x_gen.size(0)):
        #     seq = x_gen[i].cpu().numpy()
        #     seq = seq[:lengths[i]]  # обрезаем по padding
        #     if len(seq) > 0 and seq[-1] == self.generator.end_token:
        #         seq = seq[:-1]
        #     try:
        #         smi = self.get_mapped(seq.tolist())
        #     except Exception as e:
        #         smi = f"DECODE_ERROR: {e}"
        #     smiles_list.append(smi)
        
        # print("\nGenerated SMILES:")
        # for i, smi in enumerate(smiles_list):
        #     print(f"{i+1}: {smi}")
        
        # # Проверка валидности
        # valids = [Chem.MolFromSmiles(smi) is not None for smi in smiles_list]
        # print("Validity:", valids)

        batch_size, len_real = x.size()
        x_real = x.to(self.device)
        if self.label_smoothing:
            y_real = torch.full((batch_size, len_real), self.label_smoothing_params[1], device=self.device)
        else:
            y_real = torch.ones(batch_size, len_real).to(self.device)


        for i in range(self.num_gen_iterations):


            z = self.sample_latent(batch_size)
            generator_outputs = self.generator.forward(z, max_len=40)
            x_gen, log_probs, entropies = generator_outputs.values()

            # НОВАЯ СТРОЧКА  
            # x_gen_for_disc = torch.where(x_gen == -1, torch.zeros_like(x_gen), x_gen)

            _, len_gen = x_gen.size()
            # if self.label_smoothing:
            #     y_gen = torch.full((batch_size, len_gen), self.label_smoothing_params[0], device=self.device)
            # else:
            #     y_gen = torch.zeros(batch_size, len_gen).to(self.device)

            y_gen = -torch.ones(batch_size, len_gen, device=self.device)    # -1

            #####################
            # Train Discriminator
            #####################
            if i==0:
                self.discriminator_optim.zero_grad()
                fake_loss = self.discriminator_loss(x_gen, y_gen, is_real=False)
                # fake_loss = self.discriminator_loss(x_gen_for_disc, y_gen)
                real_loss = self.discriminator_loss(x_real, y_real, is_real=True)
                # discr_loss = 0.5 * (real_loss + fake_loss)

                discr_loss = fake_loss + real_loss
                discr_loss.backward()
                clip_grad_value_(self.discriminator.parameters(), 0.1)
                self.discriminator_optim.step()

            #################
            # Train Generator
            #################
            self.generator_optim.zero_grad()
            y_pred, y_pred_mask = self.discriminator(x_gen).values()
            lengths = y_pred_mask.sum(1).long()
            smiles_list = [self.get_mapped(x_i[:l-1].numpy()) for x_i, l in zip(x_gen.cpu(), lengths)]
            
            # y_pred, y_pred_mask = self.discriminator(x_gen_for_disc).values()
            # r_value = 2 * y_pred - 1
            unique_count = len(set(smiles_list))
            uniqueness = unique_count / len(smiles_list)

            r_value = 0.7 * y_pred + 0.3 * uniqueness


            if self.reward_clamp:
                R = torch.clamp(r_value, min=-0.9, max=0.9)
            else:
                 R = r_value

            if self.add_validity:
                validity_rewards = []
                for smi in smiles_list:
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            if ' ' not in smi:
                                validity_rewards.append(1.0)
                            else:
                                validity_rewards.append(0.0)
                        else:
                            validity_rewards.append(0.0)
                    except:
                        validity_rewards.append(0.0)
                validity_rewards = torch.tensor(validity_rewards, device=R.device).unsqueeze(1)
                R = (1 - self.add_validity) * R + self.add_validity * validity_rewards

            lengths = y_pred_mask.sum(1).long()
            list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]

            generator_loss = []
            for reward, log_p in zip(list_rewards, log_probs):
                reward_baseline = reward - self.b
                generator_loss.append((- reward_baseline * log_p).sum())

            generator_loss = torch.stack(generator_loss).mean() - sum(entropies) * self.entropy_weight / batch_size
            generator_loss.backward()
            clip_grad_value_(self.generator.parameters(), self.gen_clip_grad_value)
            self.generator_optim.step()

            # Update baseline
            with torch.no_grad():
                mean_reward = (R * y_pred_mask).sum() / y_pred_mask.sum()
                self.b = self.update_baseline_weights[0] * self.b + self.update_baseline_weights[1] * mean_reward  # <<< медленнее!

            # Save metrics
            loss_disc_val = discr_loss.item()
            loss_gen_val = generator_loss.item()
            mean_reward_val = mean_reward.item()

        print("Generated SMILES:")
        for i in range(min(5, len(smiles_list))):
            print(f"{i+1}: {smiles_list[i]}")

        return {
            'loss_disc': loss_disc_val,
            'loss_gen': loss_gen_val, 
            'mean_reward': mean_reward_val
        }

    def create_dataloader(self, data, batch_size=128, shuffle=True, num_workers=5):
        """create a dataloader

        Args:
            data (list[str]): list of molecule smiles
            batch_size (int, optional): Defaults to 128.
            shuffle (bool, optional): Defaults to True.
            num_workers (int, optional): Defaults to 5.

        Returns:
            torch.data.DataLoader: a torch dataloader
        """

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.tokenizer.batch_tokenize,
            num_workers=num_workers
        )

    def train_n_steps(self, train_loader, max_epoch=10000, evaluate_every=50, save_every=100):
        for epoch in range(max_epoch):
            print('#' * 12, f'Epoch: {epoch}', '#' * 12, sep='\n')
            
            epoch_metrics = {'loss_disc': [], 'loss_gen': [], 'mean_reward': []}
            
            for i, batch in enumerate(train_loader):
                print('Batch #', i)
                metrics = self.train_step(batch)
                self.global_step += 1

                # Сохраняем метрики каждого шага (опционально)
                self.history['step'].append(self.global_step)
                self.history['loss_disc'].append(metrics['loss_disc'])
                self.history['loss_gen'].append(metrics['loss_gen'])
                self.history['mean_reward'].append(metrics['mean_reward'])
                current_valid = getattr(self, '_last_valid_score', float('nan'))

                # Оценка каждые N батчей
                if i % evaluate_every == 0:
                    self.eval()
                    score = self.evaluate_n(100)
                    self.train()
                    print(f'Valid = {score:.2f}')
                    current_valid = score
                    self._last_valid_score = score
                self.history['mean_valid_100'].append(current_valid)
                
                if self.global_step % save_every == 0:
                    with open(self.model_path, 'wb') as f:
                        pi.dump(self, f)
                    print(f'Model is updated on {self.global_step} global step')


                log_file = os.path.join(self.log_path, "training_log.csv")
                pd.DataFrame(self.history).to_csv(log_file, index=False)
                # Отрисовка графиков
                plt.figure(figsize=(12, 4))
                for idx, (key, title, scale) in enumerate([
                    ('loss_disc', 'Discriminator Loss (scaled)', True),
                    ('loss_gen', 'Generator Loss (scaled)', True),
                    ('loss_gen', 'Generator Loss', False),
                    ('mean_reward', 'Mean Reward', False)
                ], 1):
                    plt.subplot(1, 4, idx)
                    plt.plot(self.history['step'], self.history[key])
                    plt.title(title)
                    plt.xlabel('Step')
                    if scale:
                        plt.ylim(-1, 1)
                    plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.log_path, "losses.png"))
                plt.close()

            # === Конец эпохи: сохранение и отрисовка ===
            
            print(f"Epoch {epoch} completed. Plot and log saved.")
    

    def get_mapped(self, seq):
        """Transform a sequence of ids to string

        Args:
            seq (list[int]): sequence of ids

        Returns:
            str: string output
        """
        return ''.join([self.tokenizer.inv_mapping[i] for i in seq])

    @torch.no_grad()
    def generate_n(self, n):
        """Generate n molecules

        Args:
            n (int)

        Returns:
            list[str]: generated molecules
        """

        z = torch.randn((n, self.hidden_dim)).to(self.device)

        x = self.generator(z)['x'].cpu()

        # НОВАЯ СТРОЧКА
        # x = torch.where(x == -1, torch.zeros_like(x), x) 

        #pd.DataFrame
        lenghts = (x > 0).sum(1)

        # l - 1 because we exclude end tokens
        return [self.get_mapped(x[:l-1].numpy()) for x, l in zip(x, lenghts)]

    def evaluate_n(self, n, path = None):
        """Evaluation: frequence of valid molecules using rdkit

        Args:
            n (int): number of sample

        Returns:
            float: requence of valid molecules
        """

        pack = self.generate_n(n)

        print(pack[:2])

        valid = np.array([Chem.MolFromSmiles(k) is not None and ' ' not in k for k in pack])
        if path is not None:
            pd.DataFrame(data={'0': pack, 'val_check': list(valid)}).to_csv(path)
        return valid.mean()