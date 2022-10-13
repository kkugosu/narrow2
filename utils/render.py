import torch
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Render:
    def __init__(self, control, policy, index, skill_num, env, key, naf):
        self.policy = policy
        self.index = index
        self.skill_num = skill_num
        self.env = env
        self.key = key

        self.control = control
        self.naf = self.control.naf_list

        # self.control.key.load_state_dict(torch.load("Parameter/narrow" + "/" + "concept" + "/" + "key"))

        i = 0
        while i < len(self.control.policy_list):
            self.control.policy_list[i].load_state_dict(torch.load("Parameter/narrow" + "/" + "SAC_conti" + "/" + "policy" + str(i)))
            self.control.upd_queue_list[i].load_state_dict(torch.load("Parameter/narrow" + "/" + "SAC_conti" + "/" + "queue" + str(i)))
            i = i + 1

        assert self.naf[0].policy is self.control.policy_list[0], "as error"

    def rend(self, traj=None):
        n_p_o = self.env.reset()
        t = 0
        total_performance = 0
        fail_time = 0
        cir = 0
        while t < self.skill_num*traj:
            # n_a = self.policy.action(n_p_o, self.index, per_one=1)
            # print(cir)
            n_a = self.policy.action(n_p_o, self.naf, index=cir, per_one=1, encoder=self.key, random=1)
            t_p_o = torch.from_numpy(n_p_o)
            t_a = torch.from_numpy(n_a)

            sa = torch.cat((t_p_o, t_a), -1).type(torch.float32)
            if traj is None:
                n_o, n_r, n_d, info = self.env.step(n_a)

            else:
                n_o, n_r, info = self.env.step(n_a.squeeze())
                print(n_a)
                print(t_p_o)
                print(self.control.upd_queue_list[0](sa))
                n_d = 0
            total_performance = total_performance + n_r
            n_p_o = n_o
            time.sleep(0.01)
            self.env.render()
            t = t + 1

            if traj is None:
                if n_d == 1:
                    print("Episode finished after {} timesteps".format(t))
                    fail_time = fail_time + 1
                    self.env.reset()
            else:
                if t % traj == 0:

                    print("Episode finished after {} timesteps".format(t))
                    fail_time = fail_time + 1
                    cir = cir + 1
                    self.env.reset()
        print("performance = ", total_performance/fail_time)
        self.env.close()
