import torch


class ExperienceBuffer:

    def __init__(self, horizon_length, num_envs, device):
        self.tensor_dict = {}
        self.horizon_length = horizon_length
        self.num_envs = num_envs
        self.device = device

    def add_buffer(self, name, shape, dtype=None):
        self.tensor_dict[name] = torch.zeros(self.horizon_length, self.num_envs, *shape, dtype=dtype, device=self.device)

    def update_data(self, name, idx, data):
        self.tensor_dict[name][idx, :] = data

    def __len__(self):
        return len(self.tensor_dict)

    def __getitem__(self, buf_name):
        return self.tensor_dict[buf_name]

    def keys(self):
        return self.tensor_dict.keys()
