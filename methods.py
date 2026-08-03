import torch
from torch.linalg import vector_norm
from distances import euclidean_distance, poincare_distance, distance_matrix

import matplotlib.pyplot as plt

EPS = 0.00001

sigma = 0.1# 0.1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# #tanh or sinh
# def exp_map(v):
#     norm = vector_norm(v, dim=-1, keepdim=True)
#     return torch.tanh(norm) * torch.div(v, norm
                                        
def exp_map(v, eps=1e-8):
    norm = vector_norm(v, dim=-1, keepdim=True)
    norm_safe = torch.clamp(norm, min=eps)
    return torch.tanh(norm_safe) * v / norm_safe

def log_map(x, eps=1e-8):
    norm = vector_norm(x, dim=-1, keepdim=True)
    norm_safe = torch.clamp(norm, min=eps, max=1 - eps)
    return torch.atanh(norm_safe) * x / norm_safe



def isomap_kernel(data_dist_matrix):  # input should be a distance matrix D
    N = data_dist_matrix.shape[0]  # number of points considered
    I = torch.eye(N).to(device)
    A = torch.ones(N, N).to(device)
    return -0.5 * torch.matmul(torch.matmul(I - (1 / N) * A, torch.matmul(data_dist_matrix, data_dist_matrix)),
                               (I - (1 / N) * A))


class Embedder(torch.nn.Module):
    # An abstract class for embedding methods.

    def __init__(self, data_size, latent_dim, latent_dist_fun=euclidean_distance, distr='gaussian'):
        super().__init__()
        if distr == 'gaussian':
            with torch.no_grad():
                self.embeddings = sigma * torch.randn((data_size, latent_dim), requires_grad=True)  # Gaussian
                # self.embeddings = torch.nn.Parameter(self.embeddings, requires_grad = True)###
                self.embeddings = torch.rand((data_size, latent_dim), requires_grad=True) #Uniform
                self.embeddings = torch.nn.Parameter(self.embeddings)

        if distr == 'hypergaussian':
            with torch.no_grad():
                self.embeddings = exp_map(sigma * torch.randn((data_size, latent_dim), requires_grad=True))

        self.latent_dist_fun = latent_dist_fun
        self.data_size = data_size

    def normalize(self, normalization=True):
        if normalization:
            with torch.no_grad():
                norms = vector_norm(self.embeddings, dim=-1).unsqueeze(-1)
                self.embeddings.data = torch.where(norms < 1, self.embeddings, self.embeddings / (norms + EPS))
                self.embeddings.requires_grad = True
                self.embeddings.retain_grad()
        else:
            with torch.no_grad():
                norms = vector_norm(self.embeddings, dim=-1).unsqueeze(-1)
                self.embeddings.data = torch.where(norms < 1, self.embeddings, self.embeddings / 1)
                self.embeddings.requires_grad = True
                self.embeddings.retain_grad()

    def forward(self):
        pass

    def loss_fun(self):
        pass


class MDS(Embedder):
    def loss_fun(self, data_dist_matrix, idx, data_binary_dist_matrix=None, temperature=None):
        latent_dist_matrix = distance_matrix(self.embeddings[idx], self.latent_dist_fun)
        return ((data_dist_matrix - latent_dist_matrix) ** 2).mean()


class Isomap(Embedder):
    def loss_fun(self, data_dist_matrix, idx, data_binary_dist_matrix=None, temperature=None):
        latent_dist_matrix = distance_matrix(self.embeddings[idx], self.latent_dist_fun)
        isomap_term = isomap_kernel(data_dist_matrix) - isomap_kernel(latent_dist_matrix)
        ##loss = torch.norm(isomap_term, p='fro')/self.data_size
        loss = torch.einsum("ij, ij ->", isomap_term, isomap_term) / self.data_size
        return loss


class Contrastive(Embedder):
    def __init__(self, data_size, latent_dim, latent_dist_fun=euclidean_distance, distr='gaussian'):
        super().__init__(data_size, latent_dim, latent_dist_fun, distr)
        self.losses_pos = []
        self.losses_neg = []

    def loss_fun(self, data_dist_matrix, idx, neighborhood_matrix, temperature=1., metricity=True):
        latent_dist_matrix = distance_matrix(self.embeddings[idx], self.latent_dist_fun)

        positive_pairs = neighborhood_matrix == 1
        positive_pairs = torch.tensor(positive_pairs)

        # pos_loss = latent_dist_matrix[positive_pairs].sum()/temperature

        # temp_pos = 1. #1/temperature
        pos_loss = ((latent_dist_matrix[positive_pairs] - float(metricity) * data_dist_matrix[
            positive_pairs]) ** 2).mean() / (temperature)  # with metricity injected

        negative_pairs = neighborhood_matrix == 0
        negative_pairs = torch.tensor(negative_pairs)
        # print(negative_pairs.shape)

        # neg_loss = torch.logsumexp(-latent_dist_matrix[negative_pairs]/temperature, dim=0)
        neg_loss = torch.logsumexp(-(latent_dist_matrix[negative_pairs]) ** 2 / temperature, dim=0).mean()
        self.losses_pos.append(pos_loss.item())
        self.losses_neg.append(neg_loss.item())
        loss = pos_loss + neg_loss

        return loss

# if __name__=='__main__':
#     X = Embedder(1000, 2, distr='hypergaussian')
#     # X.normalize()
#     pts = X.embeddings.detach().cpu().numpy()
#     plt.scatter(pts[:,0], pts[:,1])
#     plt.gca().set_aspect('equal', 'box')
#     plt.show()