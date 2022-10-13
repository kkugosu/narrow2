from policy import DDPG, SAC_conti
from utils import render, train, buffer, dataset, dataloader
from control import concept
import gym
from simple_env import narrow, plane, wallplane
from utils import converter
import torch
import viewf
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    BATCH_SIZE = 10000
    CAPACITY = 10000
    TRAIN_ITER = 100
    MEMORY_ITER = 100
    learning_rate = 0.01
    policy = None
    env = None
    env_name = None
    control = None
    control_name = None
    e_trace = 1
    precision = 3
    model_type = None

    def get_integer():
        _valid = 0
        while _valid == 0:
            integer = input("->")
            try:
                int(integer)
                if float(integer).is_integer():
                    _valid = 1
                    return int(integer)
                else:
                    print("enter integer")
            except ValueError:
                print("enter integer")

    def get_float():
        _valid = 0
        while _valid == 0:
            float_ = input("->")
            try:
                float(float_)
                _valid = 1
                return float(float_)
            except ValueError:
                print("enter float")
    print("use simple env?")
    simple = 1
    if simple == 1:
        valid = 0
        while valid == 0:
            print("enter envname, {narrow, plane, wallplane}")
            env_name = "narrow"#input("->")
            if env_name == "narrow":
                env = narrow.Narrow()
                valid = 1
            elif env_name == "plane":
                env = plane.Plane()
                valid = 1
            elif env_name == "wallplane":
                env = wallplane.WallPlane()
                valid = 1
            else:
                print("error")
    else:
        valid = 0
        while valid == 0:
            print("enter envname, {cartpole as cart, hoppper as hope}")
            env_name = input("->")
            if env_name == "cart":
                env = gym.make('CartPole-v1')
                valid = 1
                print("we can't use DDPG")
            elif env_name == "hope":
                env = gym.make('Hopper-v3')
                valid = 1
                print("enter hopper precision 3")
                precision = 3
                # precision = get_integer()
            else:
                print("error")

    STATE_LENGTH = 2

    if env_name == "cart":
        ACTION_LENGTH = 2
        A_index_L = 2
    elif env_name == "plane":
        ACTION_LENGTH = 2
        A_index_L = 2
    elif env_name == "narrow":
        ACTION_LENGTH = 2
        A_index_L = 2
    elif env_name == "wallplane":
        ACTION_LENGTH = 2
        A_index_L = 2
    else:
        ACTION_LENGTH = len(env.action_space.sample())
        A_index_L = precision ** ACTION_LENGTH

    """
    valid = 0
    while valid == 0:
        print("model_free : 0, model_based : 1, meta : 2")
        print(" meta ")
        model_type = get_integer()
        if (model_type >= 0) | (model_type < 3):
            valid = 1
            
    """

    # print("enter batchsize recommend 1000")
    # BATCH_SIZE = get_integer()

    print("enter memory capacity recommend 1000")
    print("batchsize = capacity")
    print("capacity = 1000")
    CAPACITY = 1500 # get_integer()
    BATCH_SIZE = CAPACITY

    print("memory reset time recommend 100")
    print("train iter = 100")
    TRAIN_ITER = 10

    print("train_iteration per memory recommend 10")
    print("memory iter = 10")
    MEMORY_ITER = 1

    print("enter learning rate recommend 0.01")
    print("learning rate = 0.0001")
    learning_rate = 1e-6

    print("enter eligibility trace step, if pg: 100, if gps: 1")
    print("e_trace = 1")
    e_trace = 1

    print("done penalty, if cartpole, recommend 10")
    print("done penalty = 1")
    done_penalty = 0

    print("load previous model 0 or 1")
    load_ = 0 #input("->")

    print("num_skills?")
    print("skillnum = 10")
    skill_num = 1

    my_converter = converter.IndexAct(env_name, ACTION_LENGTH, precision, BATCH_SIZE)
    print("encode_state = 0")
    encode_state = 0
    arg_list = [env, learning_rate, skill_num, MEMORY_ITER, STATE_LENGTH, ACTION_LENGTH,
                A_index_L, my_converter, encode_state, DEVICE]
    model_type = 1
    data = dataset.SimData(capacity=CAPACITY)
    _dataloader = dataloader.CustomDataLoader(data, batch_size=BATCH_SIZE)
    my_buffer = buffer.Memory(env, step_size=e_trace, done_penalty=done_penalty, skill_num=skill_num,
                              capacity=CAPACITY, dataset=data, dataloader=_dataloader)
    if model_type == 1:
        valid = 0
        while valid == 0:
            print("enter RL policy, {DDPG, SAC_conti}")
            policy_n = "SAC_conti" # input("->")
            if policy_n == "DDPG":
                policy = DDPG.DDPGPolicy(*arg_list)
                valid = 1
            elif policy_n == "SAC_conti":
                policy = SAC_conti.SACPolicy(*arg_list)
                valid = 1
            else:
                print("error")
    elif model_type == 2:
        valid = 0
        while valid == 0:
            print("enter RL policy, {gps}")
            control = input("->")
            if control == "gps":
                policy = gps.GPS(*arg_list)
                valid = 1
            else:
                print("error")
    else:
        print("error")
        
    valid = 0
    while valid == 0:
        print("enter RL control, {concept}")
        control_name = "concept" #input("->")
        if control_name == "concept":
            control = concept.Concept(my_buffer, learning_rate, STATE_LENGTH, ACTION_LENGTH, policy, skill_num, DEVICE)
            valid = 1
        else:
            print("control name error")

    my_train = train.Train(TRAIN_ITER, MEMORY_ITER, skill_num,
                           CAPACITY, env, control, env_name, load_)
    print("pre train")
    naf = my_train.train_skill_simultaneously()
    print("train")
    # index = my_train.simulate()
    print("rendering")
    # my_train.post_train(index)
    # viewf.view(encoder, encoder)
    my_rend = render.Render(control, policy, 0, skill_num, env, key=None, naf=naf)
    my_rend.rend(CAPACITY/skill_num)
