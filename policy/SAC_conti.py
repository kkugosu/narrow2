from policy import BASE
import torch
import numpy as np
from torch import nn
from NeuralNetwork import basic_nn
from utils import converter

GAMMA = 0.98
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SACPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.criterion = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def action(self, n_s, policy, index, per_one=1, encoder=None):
        t_s = torch.from_numpy(n_s).type(torch.float32).to(self.device)
        # t_s = state_converter(t_s)
        if encoder is None:
            pass
        else:
            t_s = encoder(t_s)
        with torch.no_grad():
            mean, v, t_a = policy[index].prob(t_s)
            t_a = torch.clamp(t_a, min=-10, max=10)
        n_a = t_a.cpu().numpy()
        n_a = n_a
        return n_a

    def update(self, *trajectory, cal_reward, policy_list, naf_list, upd_queue_list, base_queue_list,
               optimizer_p, optimizer_q, memory_iter=0, encoder=None):
        i = 0
        queue_loss = None
        policy_loss = None
        while i < self.sk_n:
            base_queue_list[i].load_state_dict(upd_queue_list[i].state_dict())
            base_queue_list[i].eval()
            i = i + 1
        i = 0
        if memory_iter != 0:
            self.m_i = memory_iter
        else:
            self.m_i = 1
        while i < self.m_i:

            n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
            t_p_s = torch.tensor(n_p_s, dtype=torch.float32).to(self.device)
            # t_p_s = batch_state_converter(t_p_s)
            if encoder is not None:
                with torch.no_grad():
                    encoded_t_p_s = encoder(t_p_s)
            else:
                encoded_t_p_s = t_p_s
            t_p_s = self.skill_converter(encoded_t_p_s, sk_idx, per_one=0)
            t_a = torch.tensor(n_a, dtype=torch.float32).to(self.device)
            t_a = self.skill_converter(t_a, sk_idx, per_one=0)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)

            t_r_u = t_r.unsqueeze(-1)
            t_r = self.skill_converter(t_r_u, sk_idx, per_one=0).squeeze()
            t_r = t_r.unsqueeze(0)
            # policy_loss = torch.mean(torch.log(t_p_weight) - t_p_qvalue)
            # we already sampled according to policy

            policy_loss = torch.tensor(0).to(self.device).type(torch.float32)

            skill_id = 0  # seq training
            while skill_id < self.sk_n:
                _ts = torch.zeros((3, len(n_s), 2))
                mean, _, _ = naf_list[skill_id].prob(t_p_s[skill_id])
                _nps = t_p_s[skill_id].cpu().numpy()
                action = -np.ones(len(_nps))
                n_s = self.env.pseudo_step(_nps, action)
                _ts[0] = torch.from_numpy(n_s).to(DEVICE)
                action = np.zeros(len(_nps))
                n_s = self.env.pseudo_step(_nps, action)
                _ts[1] = torch.from_numpy(n_s).to(DEVICE)
                action = np.ones(len(_nps))
                n_s = self.env.pseudo_step(_nps, action)
                _ts[2] = torch.from_numpy(n_s).to(DEVICE)

                target_0 = cal_reward(t_p_s[skill_id], _ts[0], self.sk_n)
                target_0 = target_0 - cal_reward(t_p_s[skill_id], t_p_s[skill_id], self.sk_n)
                target_1 = cal_reward(t_p_s[skill_id], _ts[1], self.sk_n)
                target_1 = target_1 - cal_reward(t_p_s[skill_id], t_p_s[skill_id], self.sk_n)
                target_2 = cal_reward(t_p_s[skill_id], _ts[1], self.sk_n)
                target_2 = target_2 - cal_reward(t_p_s[skill_id], t_p_s[skill_id], self.sk_n)
                target = torch.cat((target_0, target_1), -1)
                target = torch.cat((target, target_2), -1)
                # target size len, 3

                x = torch.tensor([-1, 0, 1])
                x = x.repeat(len(_nps))
                diff = (x - mean.squeeze())
                prob = (-1 / 2) * torch.square(diff)
                policy_loss = torch.sum(torch.exp(prob) * (prob - target))

                skill_id = skill_id + 1

            policy_loss = policy_loss

            sa_pair = torch.cat((t_p_s, t_a), -1).type(torch.float32)
            skill_id = 0 # seq training
            queue_loss = 0
            while skill_id < self.sk_n:
                t_p_qvalue = upd_queue_list[skill_id](sa_pair[skill_id]).squeeze()
                act = naf_list[skill_id]()
                sa_pair = torch.cat((t_s, act), -1).type(torch.float32)
                t_qvalue = t_r[skill_id] + GAMMA*base_queue_list[skill_id]().squeeze()

                queue_loss = queue_loss + self.criterion(t_p_qvalue, t_qvalue)
                skill_id = skill_id + 1
            print("queueloss = ", queue_loss)

            print("policy loss = ", policy_loss)

            # print("queue_loss = ", queue_loss)
            optimizer_p.zero_grad()
            policy_loss.backward(retain_graph=True)

            i = 0  # seq training
            while i < len(policy_list):
                for param in policy_list[i].parameters():
                    param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
                    param.grad.data.clamp_(-1, 1)
                i = i + 1
            optimizer_p.step()

            optimizer_q.zero_grad()
            queue_loss.backward()
            i = 0 # seq training
            while i < len(upd_queue_list):
                for param in upd_queue_list[i].parameters():
                    param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
                    param.grad.data.clamp_(-1, 1)
                i = i + 1
            if torch.isnan(queue_loss):
                pass
            else:
                optimizer_q.step()

            i = i + 1
        # print("loss1 = ", policy_loss.squeeze())
        # print("loss2 = ", queue_loss.squeeze())

        # return torch.stack((policy_loss.squeeze(), queue_loss.squeeze()))
        return policy_loss.squeeze()