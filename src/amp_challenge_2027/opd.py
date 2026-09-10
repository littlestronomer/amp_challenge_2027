"""On-policy token distillation and a separate moment-coverage policy gradient."""
import numpy as np

from amp_challenge_2027.config import PAD_ID


def forward_kl(teacher, student, tokens):
    """Mean sequence-normalized KL(q_teacher || p_student), excluding PAD."""
    mask = tokens[:, 1:] != PAD_ID
    values = (teacher.detach().exp() * (teacher.detach()-student)).sum(-1)
    return ((values*mask).sum(1)/mask.sum(1)).mean()


def coverage_cost(features, target):
    """Leave-one-out costs for the gradient of squared mean-feature discrepancy.

    The self-pair diagonal is excluded. Do not group-center these costs: a
    shared batch penalty would vanish in a group-relative reward objective.
    """
    phi = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (phi.ndim != 2 or len(phi) < 2 or target.shape != (phi.shape[1],) or
            not np.isfinite(phi).all() or not np.isfinite(target).all()):
        raise ValueError("Invalid coverage features")
    others = (phi.sum(0)-phi)/(len(phi)-1)
    costs = 2*np.sum(phi*(others-target), axis=1)
    discrepancy = float(np.mean(np.sum(phi*others, axis=1))-2*phi.mean(0).dot(target)+target.dot(target))
    return costs.astype(np.float32), discrepancy


def coverage_loss(student, tokens, costs):
    """One on-policy REINFORCE update; full sequence log probability, not /length."""
    action = tokens[:, 1:]
    mask = action != PAD_ID
    logp = student.gather(-1, action[..., None]).squeeze(-1)
    return ((logp*mask).sum(1)*costs.detach()).mean()


class CoverageFeatures:
    """Fixed RBF random features of unit-norm35M embeddings plus length/AAC."""
    def __init__(self, encoder, sequences, seed):
        self.encoder = encoder
        z = self.normalized(sequences)
        if z.shape[1] != 480:
            raise ValueError("Coverage protocol requires the pinned35M, 480-dimensional encoder")
        sample = z[:min(256, len(z))]
        distances = np.sqrt(np.maximum(0., ((sample[:, None]-sample[None, :])**2).sum(-1)))
        positive = distances[distances > 1e-8]
        self.bandwidth = float(np.median(positive)) if len(positive) else 1.
        random = np.random.default_rng(seed)
        self.projection = random.normal(size=(480, 128))/self.bandwidth
        self.phase = random.uniform(0, 2*np.pi, 128)
        self.target = self.transform(sequences, z).mean(0)

    def normalized(self, sequences):
        z = np.asarray(self.encoder.encode(sequences, 64), dtype=np.float64)
        if z.ndim != 2 or len(z) != len(sequences) or not np.isfinite(z).all():
            raise ValueError("Invalid coverage embeddings")
        return z/np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

    def transform(self, sequences, z=None):
        if z is None:
            z = self.normalized(sequences)
        random_features = np.sqrt(2/128)*np.cos(z@self.projection+self.phase)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        descriptors = np.array([[len(s)/50, *[s.count(a)/len(s) for a in amino_acids]] for s in sequences])
        return np.concatenate((random_features, descriptors), axis=1)
