import torch


class EmbedOptim():
    # An abstract class for optimizers for embedding methods

    def __init__(self, embedder, lr):
        # embedder: Embedder , lr: Float

        self.embedder = embedder
        self.lr = lr

    def zero_grad(self):
        self.embedder.embeddings.grad = torch.zeros_like(self.embedder.embeddings)

    def step(self):
        pass


class StandardOptim(EmbedOptim):
    def step(self, idx):
        with torch.no_grad():
            embeddings = self.embedder.embeddings
            # print(embeddings.grad)
            # embeddings[idx] -= self.lr * self.embedder.embeddings.grad[idx]

            grad = self.embedder.embeddings.grad.detach()
            embeddings[idx] -= self.lr * grad[idx]


class AdamOptim(EmbedOptim):
    def __init__(self, embedder, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(embedder,lr)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize moment estimates for each parameter
        self.m = torch.zeros_like(embedder.embeddings)
        self.v = torch.zeros_like(embedder.embeddings)
        self.t = 0

    def step(self, idx):
        with torch.no_grad():
            # Increment time step
            self.t += 1

            # Get the gradient of the embeddings
            grad = self.embedder.embeddings.grad.detach()

            # Compute biased first moment estimate
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad[idx]

            # Compute biased second moment estimate
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad[idx] ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second moment estimate
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            # Update the embeddings using the Adam rule
            self.embedder.embeddings[idx] -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

class PoincareOptim(EmbedOptim):
    def step(self, idx):
        with torch.no_grad():
            embeddings = self.embedder.embeddings
            norms = (embeddings ** 2).sum(-1).unsqueeze(-1)
            embeddings[idx] -= self.lr * (1 - norms[idx]) ** 2 * embeddings.grad[idx] / 4


class PoincareRiemannianAdamOptim(EmbedOptim):

    def __init__(
        self,
        embedder,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        amsgrad=False,
        proj_eps=1e-5,
    ):
        super().__init__(embedder, lr)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.proj_eps = proj_eps

        # First moment: tangent vector, same shape as embeddings (N, 2)
        self.m = torch.zeros_like(embedder.embeddings)
        # Second moment: one scalar per embedding point, shape (N, 1)
        self.v = torch.zeros((*embedder.embeddings.shape[:-1], 1), dtype=embedder.embeddings.dtype,
                             device=embedder.embeddings.device)
        self.v_max = torch.zeros_like(self.v) if amsgrad else None
        self.t = 0

    @staticmethod
    def _lambda_x(x):
        return 2.0 / (1.0 - (x ** 2).sum(dim=-1, keepdim=True)).clamp_min(1e-15)

    def _project(self, x):
        maxnorm = 1.0 - self.proj_eps
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-15)
        scale = torch.where(norm > maxnorm, maxnorm / norm, torch.ones_like(norm))
        return x * scale

    @staticmethod
    def _mobius_add(x, y):
        x2 = (x ** 2).sum(dim=-1, keepdim=True)
        y2 = (y ** 2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        den = 1 + 2 * xy + x2 * y2
        return num / den.clamp_min(1e-15)

    def _expmap(self, x, v):
        v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-15)
        lam = self._lambda_x(x)
        second_term = torch.tanh(lam * v_norm / 2.0) * v / v_norm
        y = self._mobius_add(x, second_term)
        return self._project(y)

    @staticmethod
    def _gyration(u, v, w):
        u2 = (u ** 2).sum(dim=-1, keepdim=True)
        v2 = (v ** 2).sum(dim=-1, keepdim=True)
        uv = (u * v).sum(dim=-1, keepdim=True)
        uw = (u * w).sum(dim=-1, keepdim=True)
        vw = (v * w).sum(dim=-1, keepdim=True)
        a = -uw * v2 + vw + 2 * uv * vw
        b = -vw * u2 - uw
        d = 1 + 2 * uv + u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(1e-15)

    def _transport(self, x, y, v):
        lam_x = self._lambda_x(x)
        lam_y = self._lambda_x(y)
        return self._gyration(y, -x, v) * (lam_x / lam_y)

    def _egrad_to_rgrad(self, x, grad):
        lam = self._lambda_x(x)
        return grad / (lam ** 2)

    def _rgrad_norm_sq(self, x, rgrad):
        lam = self._lambda_x(x)
        return (lam ** 2) * (rgrad ** 2).sum(dim=-1, keepdim=True)

    def step(self, idx):
        with torch.no_grad():
            self.t += 1

            embeddings = self.embedder.embeddings
            grad = embeddings.grad.detach()

            # Keep only valid Poincare points before updating.
            embeddings.data = self._project(embeddings.data)

            x = embeddings[idx]
            grad_euc = grad[idx]

            # Convert Euclidean gradient to Riemannian gradient.
            grad_r = self._egrad_to_rgrad(x, grad_euc)

            # First moment (momentum) in tangent space.
            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * grad_r

            # Second moment: one scalar per point using the Riemannian norm squared.
            grad_norm_sq = self._rgrad_norm_sq(x, grad_r)
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * grad_norm_sq

            # Bias corrections.
            m_hat = self.m[idx] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1.0 - self.beta2 ** self.t)

            if self.amsgrad:
                self.v_max[idx] = torch.maximum(self.v_max[idx], v_hat)
                denom = torch.sqrt(self.v_max[idx]) + self.epsilon
            else:
                denom = torch.sqrt(v_hat) + self.epsilon

            # Tangent update direction.
            direction = -self.lr * m_hat / denom

            # Move on the manifold with the exponential map.
            new_x = self._expmap(x, direction)

            # Transport momentum to the new tangent space.
            self.m[idx] = self._transport(x, new_x, self.m[idx])

            embeddings[idx] = new_x
            embeddings.data = self._project(embeddings.data)

