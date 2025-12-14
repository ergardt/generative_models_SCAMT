import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch import nn
import torch.utils.data as torch_data
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pickle as pi

from scripts.layers import Generator, RecurrentDiscriminator, TransformerDiscriminator, TransformerGenerator
from scripts.tokenizer import Tokenizer


class MolGen(nn.Module):

    def __init__(self, 
                 data, 
                 hidden_dim=64, 
                 lr_optim=1e-3, 
                 lr_discr=1e-3, 
                 log_path =  './molgen_logs/tmp',
                #  label_smoothing = False,
                #  label_smoothing_params = [0, 1],
                 num_gen_iterations = 1,
                #  reward_clamp = False,
                #  update_baseline_weights = [0.9, 0.1],
                #  entropy_weight = 0.01,
                 gen_clip_grad_value=0.1,
                #  add_validity=False,
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
        self.max_seq_len = 100

        self.tokenizer = Tokenizer(data, max_len=self.max_seq_len)
        # self.generator = Generator(
        #     latent_dim=hidden_dim,
        #     vocab_size=self.tokenizer.vocab_size,
        #     start_token=self.tokenizer.start_token,  # no need token
        #     end_token=self.tokenizer.end_token,
        # ).to(device)
        self.generator = TransformerGenerator(
            latent_dim=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            max_len=self.max_seq_len,
            start_token=self.tokenizer.start_token,
            end_token=self.tokenizer.end_token,
            num_layers=2,
            nhead=8,
            dropout=0.1
        ).to(device)

        # self.discriminator = RecurrentDiscriminator(
        #     hidden_size=hidden_dim,
        #     vocab_size=self.tokenizer.vocab_size,
        #     start_token=self.tokenizer.start_token,
        #     bidirectional=True
        # ).to(device)

        self.discriminator = TransformerDiscriminator(
            hidden_size=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            max_len=self.max_seq_len,
            start_token=self.tokenizer.start_token,
            num_layers=2,
            nhead=8,
            dropout=0.1
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
                        'mean_valid_100': [],
                        'd_real': [],               # ← НОВОЕ
                        'd_fake': [],               # ← НОВОЕ
                        'gp': []                    # ← НОВОЕ
                        }
        self.global_step = 0
        self.num_gen_iterations =  num_gen_iterations
        # self.label_smoothing = label_smoothing
        # self.label_smoothing_params = label_smoothing_params
       
        # self.reward_clamp = reward_clamp
        # self.update_baseline_weights = update_baseline_weights
        # self.entropy_weight = entropy_weight
        self.gen_clip_grad_value = gen_clip_grad_value
        # self.add_validity=add_validity
        self.model_path = model_path

        print(f"Max sequence length in data: {self.max_seq_len}")



    def sample_latent(self, batch_size):
        """Sample from latent space

        Args:
            batch_size (int): number of samples

        Returns:
            torch.Tensor: [batch_size, self.hidden_dim]
        """
        return torch.randn(batch_size, self.hidden_dim).to(self.device)


    def gradient_penalty(self, real_tokens, fake_tokens):
        real_emb = self.discriminator.embedding(real_tokens)
        fake_emb = self.discriminator.embedding(fake_tokens)
        
        batch_size, seq_len, emb_dim = real_emb.shape
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        interpolates = alpha * real_emb + (1 - alpha) * fake_emb
        interpolates = interpolates.requires_grad_(True)
        
        # Теперь можно вызывать напрямую — без CuDNN проблем!
        disc_interpolates = self.discriminator(interpolates, from_embeddings=True)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def train_step(self, x):
        batch_size = x.size(0)
        x_real = x.to(self.device)

        # Генерация
        z = self.sample_latent(batch_size)
        gen_out = self.generator(z, max_len=x_real.size(1))
        x_gen = gen_out['x']

        # Выравнивание длины
        if x_gen.size(1) < x_real.size(1):
            pad = torch.zeros(x_gen.size(0), x_real.size(1) - x_gen.size(1), dtype=x_gen.dtype, device=x_gen.device)
            x_gen = torch.cat([x_gen, pad], dim=1)
        elif x_gen.size(1) > x_real.size(1):
            x_gen = x_gen[:, :x_real.size(1)]

        # Обучение дискриминатора
        self.discriminator_optim.zero_grad()
        real_score = self.discriminator(x_real).mean()
        fake_score = self.discriminator(x_gen).mean()
        gp = self.gradient_penalty(x_real, x_gen)  # ← добавьте это
        print(f"[Step {self.global_step}] GP: {gp.item():.6f}")
        disc_loss = -real_score + fake_score + 10.0 * gp  # ← + GP
        disc_loss.backward()
        # clip_grad_value_(self.discriminator.parameters(), 0.1)
        clip_grad_norm_(self.discriminator.parameters(), 1)

        self.discriminator_optim.step()

        # Weight clipping
        # for p in self.discriminator.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        # Обучение генератора
        gen_loss = 0.0
        if self.global_step % self.num_gen_iterations == 0:
            self.generator_optim.zero_grad()
            gen_loss = -self.discriminator(x_gen).mean()
            gen_loss.backward()
            clip_grad_value_(self.generator.parameters(), self.gen_clip_grad_value)
            self.generator_optim.step()
        else:
            gen_loss = torch.tensor(0.0, device=self.device)

        print(f"[Step {self.global_step}] D(real): {real_score.item():.4f}, D(fake): {fake_score.item():.4f}")

        return {
            'loss_disc': disc_loss.item(),
            'loss_gen': gen_loss.item(),
            'mean_reward': fake_score.item(),
            'd_real': real_score.item(),      # ← ДОБАВЛЕНО
            'd_fake': fake_score.item(),      # ← ДОБАВЛЕНО
            'gp': gp.item() 
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
                self.history['d_real'].append(metrics['d_real'])      # ← НОВОЕ
                self.history['d_fake'].append(metrics['d_fake'])      # ← НОВОЕ
                self.history['gp'].append(metrics['gp'])      
                current_valid = getattr(self, '_last_valid_score', float('nan'))

                # Оценка каждые N батчей
                if i % evaluate_every == 0:
                    self.eval()
                    score = self.evaluate_n(1000)
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
                # plt.figure(figsize=(12, 4))
                # for idx, (key, title, scale) in enumerate([
                #     ('loss_disc', 'Discriminator Loss', False),
                #     ('loss_gen', 'Generator Loss (scaled)', True),
                #     ('loss_gen', 'Generator Loss', False),
                #     ('mean_reward', 'Mean Reward', False)
                # ], 1):
                #     plt.subplot(1, 4, idx)
                #     plt.plot(self.history['step'], self.history[key])
                #     plt.title(title)
                #     plt.xlabel('Step')
                #     if scale:
                #         plt.ylim(-1, 1)
                #     plt.grid(True)
                # plt.tight_layout()
                # plt.savefig(os.path.join(self.log_path, "losses.png"))
                # plt.close()
                plt.figure(figsize=(15, 10))

                # 1. Лоссы
                plt.subplot(2, 3, 1)
                plt.plot(self.history['step'], self.history['loss_disc'])
                plt.title('Discriminator Loss')
                plt.xlabel('Step')
                plt.grid(True)

                plt.subplot(2, 3, 2)
                plt.plot(self.history['step'], self.history['loss_gen'])
                plt.title('Generator Loss')
                plt.xlabel('Step')
                plt.grid(True)

                # 2. D(real) и D(fake)
                plt.subplot(2, 3, 3)
                plt.plot(self.history['step'], self.history['d_real'], label='D(real)')
                plt.plot(self.history['step'], self.history['d_fake'], label='D(fake)')
                plt.title('Critic Scores')
                plt.xlabel('Step')
                plt.legend()
                plt.grid(True)

                # 3. Mean Reward и GP
                plt.subplot(2, 3, 4)
                plt.plot(self.history['step'], self.history['mean_reward'])
                plt.title('Mean Reward (D(fake))')
                plt.xlabel('Step')
                plt.grid(True)

                plt.subplot(2, 3, 5)
                plt.plot(self.history['step'], self.history['gp'])
                plt.title('Gradient Penalty')
                plt.xlabel('Step')
                plt.grid(True)

                # 4. Валидность
                plt.subplot(2, 3, 6)
                plt.plot(self.history['step'], self.history['mean_valid_100'])
                plt.title('Validity (last 100)')
                plt.xlabel('Step')
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
        z = torch.randn((n, self.hidden_dim)).to(self.device)
        x = self.generator(z, max_len=self.max_seq_len)['x'].cpu()
        lengths = (x > 0).sum(1)

        smiles_list = []
        for seq, l in zip(x, lengths):
            seq = seq[:l].numpy()
            if len(seq) > 0 and seq[-1] == self.generator.end_token:
                seq = seq[:-1]
            try:
                smi = self.get_mapped(seq.tolist())
            except:
                smi = ""
            smiles_list.append(smi)
        return smiles_list

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