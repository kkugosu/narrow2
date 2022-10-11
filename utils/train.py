
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, train_iter, memory_iter, skill_n,
                 capacity, env, cont, env_n, load_):

        self.t_i = train_iter
        self.m_i = memory_iter
        self.capacity = capacity
        self.cont = cont
        self.env = env
        self.skill_num = skill_n
        self.load = load_

        self.writer = SummaryWriter('Result/' + env_n + self.cont.name())

        self.PARAM_PATH = 'Parameter/' + env_n
        print("parameter path is " + self.PARAM_PATH)

        self.PARAM_PATH_TEST = 'Parameter/' + env_n + '_test'
        print("tmp parameter path is " + self.PARAM_PATH_TEST)

    def train_skill_sequentially(self):
        if self.load == 1:
            self.cont.load_model(self.PARAM_PATH)
        i = 0
        while i < self.skill_num:
            i = i + 1
            j = 0
            while j < self.t_i:
                loss = self.cont.update(self.m_i, i, self.capacity / self.skill_num)
                print("loss = ", loss)
                k = 0
                while k < len(loss):
                    self.writer.add_scalar("loss " + str(j), loss[j], i)
                    k = k + 1
                self.writer.add_scalar("performance", self.cont.get_performance(), i)
                model = self.cont.save_model(self.PARAM_PATH)

                j = j + 1

        self.writer.flush()
        self.writer.close()

    def train_skill_simultaneously(self):
        if self.load == 1:
            self.cont.load_model(self.PARAM_PATH)
        i = 0
        if i == 0:
            self.cont.save_model(self.PARAM_PATH + "1")
        while i < self.t_i:
            print(i)
            if i == 10:
                self.cont.save_model(self.PARAM_PATH + "2")
            if i == 100:
                self.cont.save_model(self.PARAM_PATH + "3")
            if i == 300:
                self.cont.save_model(self.PARAM_PATH + "4")

            print("traj = ", self.capacity / self.skill_num)
            loss, naf = self.cont.update(self.m_i, self.skill_num, self.capacity / self.skill_num)
            print("loss = ", loss)
            j = 0
            while j < len(loss):
                self.writer.add_scalar("loss " + str(j), loss[j], i)
                j = j + 1
            self.writer.add_scalar("performance", self.cont.get_performance(), i)
            model = self.cont.save_model(self.PARAM_PATH)
            i = i + 1
        self.writer.flush()
        self.writer.close()
        return naf

    def simulate(self):
        i = 0
        pre_performance = 0
        maxp_index = 0
        self.cont.load_model(self.PARAM_PATH)
        while i < self.skill_num:
            performance = self.cont.simulate(index=i, total=self.skill_num, pretrain=0, traj=self.capacity/self.skill_num)
            # print(performance)
            if performance > pre_performance:
                maxp_index = i
                pre_performance = performance
            i = i + 1
        print("max = ", maxp_index)
        print("select complete")
        return maxp_index

    def post_train(self, maxp_index):
        model = None
        i = 0
        while i < self.t_i:
            print(i)
            i = i + 1
            self.cont.simulate(index=maxp_index, total=self.skill_num, pretrain=0, traj_l=self.capacity/self.skill_num)
            loss = self.cont.update(self.m_i)
            print("loss = ", loss)
            j = 0
            while j < len(loss):
                self.writer.add_scalar("loss " + str(j), loss[j], i)
                j = j + 1
            self.writer.add_scalar("performance", self.cont.get_performance(), i)
            model = self.cont.save_model(self.PARAM_PATH)

        i = 0

        while i < len(model):

            for param in model[i].parameters():
                print("-----------" + str(i) + "-------------")
                print(param)
            i = i + 1

        self.writer.flush()
        self.writer.close()

        return maxp_index

