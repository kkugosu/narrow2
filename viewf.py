import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

x = torch.arange(5) * 200 - 400
y = torch.arange(5) * 200 - 400
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
xy = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), -1).to(DEVICE)
inp = xy.reshape(-1, 2)
print(xy.reshape(-1, 2))

x = torch.arange(9) * 100 - 400
y = torch.arange(9) * 100 - 400
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
xy = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), -1).to(DEVICE)

out = torch.exp(-torch.sum(torch.square(xy.unsqueeze(0) - inp.unsqueeze(1).unsqueeze(1))/10000, -1))
print("outsize")
print(out.size())
# out.view(-1).size()
inp = out.reshape(-1, 81).squeeze()
print(inp.size())


def view(key, query):
    inp_1 = key(inp)
    inp_2 = query(inp)
    distance_mat = torch.sum(torch.square(inp_1.unsqueeze(1) - inp_2.unsqueeze(0)), -1).reshape(25, 5, 5)[12]
    print(distance_mat)

