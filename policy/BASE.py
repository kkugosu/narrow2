import torch
import numpy as np
from abc import *


class BasePolicy(metaclass=ABCMeta):
    """
    l_r learning rate
    sk_n = skill num
    m_i memory iteration
    s_l state length
    a_l action length
    a_index_l action index length
    converter
    device
    """
    def __init__(self,
                 env,
                 l_r,
                 sk_n,
                 m_i,
                 s_l,
                 a_l,
                 a_index_l,
                 _converter,
                 encode_state,
                 device
                 ):
        self.env = env
        self.l_r = l_r
        self.sk_n = sk_n
        self.m_i = m_i
        if encode_state == 1:
            self.s_l = s_l*sk_n
        else:
            self.s_l = s_l
        self.a_l = a_l
        self.a_index_l = a_index_l
        self.converter = _converter
        self.device = device

    def skill_converter(self, t_p_o, index, per_one):

        tmp_t_p_o = torch.zeros((self.sk_n, int(len(t_p_o)/self.sk_n), len(t_p_o[0]))).to(self.device)

        i = 0
        while i < self.sk_n:

            tmp_t_p_o[i] = t_p_o[int(i * len(t_p_o)/self.sk_n):int((i + 1) * len(t_p_o)/self.sk_n)]
            i = i + 1
        t_p_o = tmp_t_p_o
        return t_p_o

    def skill_converter_one(self, t_p_o, index, per_one):

        tmp_t_p_o = torch.zeros((self.sk_n, len(t_p_o)/self.sk_n, len(t_p_o[0]))).to(self.device)

        i = 0
        while i < self.sk_n:
            tmp_t_p_o[i] = t_p_o[i * len(t_p_o)/self.sk_n:(i + 1) * len(t_p_o)/self.sk_n]
            i = i + 1
        t_p_o = tmp_t_p_o
        return t_p_o

