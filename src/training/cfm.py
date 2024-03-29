# a class for conditional flow matching methods from https://github.com/atong01/conditional-flow-matching
# modified to implement  "Flow Matching" (Lipman et al. 2023)
# we also provide a wrapper for the model to only evolve the first part of the input, taking into account the context

import math
import torch
import torch.nn as nn


import torch.nn as nn
import torch

import math
import torch

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdiffeq import odeint


class ModelWrapper(nn.Module):
    def __init__(self, base_model, max_len=10, device="cpu"):
        """
        Wraps a base model to only evolve the first part of the input specifying a certain context using the model.

        Args:
            base_model (nn.Module): The base model to wrap.
        """
        super(ModelWrapper, self).__init__()
        self.base_model = base_model.eval()
        self.device = device
        self.feature_size = base_model.feature_size
        self.max_len = max_len

    def forward(self, t, x, pu, nfakes, key_padding_mask, **kwargs):
        """
        Forward pass of the wrapped model.

        Args:
            t (torch.Tensor): The time tensor.
            x (torch.Tensor): The input tensor: concatenation of [actual input, padded pu, padded nfakes].
            shape is [N, len+2, 39]
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """

        key_padding_mask = key_padding_mask.bool()
        # Only evolve xt using the model
        t_broadcasted = t.expand(x.shape[0], 1)
        dxt_dt = self.base_model(x, pu, nfakes, t_broadcasted, key_padding_mask)

        dx_dt = dxt_dt # zeros_for_context

        return dx_dt
    
    def generate(self, pu, nfakes, key_padding_mask, t_span, solver="euler"):
        self.base_model.eval()

        initial_conditions = torch.randn(len(pu), self.max_len, self.feature_size)
        # pu_zeros = torch.zeros_like(x)
        # nfakes_zeros = torch.zeros_like(x)
        # pu_zeros[:, :, 0] = pu
        # nfakes_zeros[:, :, 0] = nfakes
        # initial_conditions = torch.cat([x, pu_zeros, nfakes_zeros, key_padding_mask], dim=2)
        initial_conditions = initial_conditions.to(self.device)
        pu = pu.to(self.device)
        nfakes = nfakes.to(self.device)
        key_padding_mask = key_padding_mask.to(self.device)

        sample = odeint(lambda t, x: self(t, x, pu, nfakes, key_padding_mask),
                        initial_conditions,
                        t_span,
                        atol=1e-6,
                        rtol=1e-6,
                        method=solver,
                        )[-1, :, :]
        # print(sample.shape)
        return sample
        


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, float):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

    

class AlphaTConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    A class representing an AlphaTConditionalFlowMatcher.

    This class extends the ConditionalFlowMatcher class and implements the
    sample_location_and_conditional_flow method to compute the sample xt and
    the conditional vector field ut(x1|x0).

    Parameters
    ----------
    sigma : float
        The standard deviation parameter for the probability distribution.
    alpha : float
        The alpha parameter for computing the sample t.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    alpha : float
        The alpha parameter for computing the sample t.

    Methods
    -------
    sample_location_and_conditional_flow(x0, x1, return_noise=False)
        Compute the sample xt and the conditional vector field ut(x1|x0).

    """

    def __init__(self, sigma, alpha, **kwargs):
        super().__init__(sigma, **kwargs)
        self.alpha = alpha

    def sample_location_and_conditional_flow(self, x0, x1, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
            The sample t.
        xt : Tensor, shape (bs, *dim)
            The samples drawn from probability path pt.
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim)
            The noise sample epsilon such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = torch.pow(torch.rand(x0.shape[0]), 1/(1+self.alpha)).type_as(x0)
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut


class MyTargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """Lipman et al. 2023 style target OT conditional flow matching. This class inherits the
    ConditionalFlowMatcher and override the compute_mu_t, compute_sigma_t and
    compute_conditional_flow functions in order to compute [2]'s flow matching.

    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path tx1, see (Eq.20) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t x1, 1 - (1 - sigma) t), see (Eq.20) [2].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma 1 - (1 - sigma) t

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t), see Eq.(21) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t)

        References
        ----------
        [1] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)
    

class MyAlphaTTargetConditionalFlowMatcher(AlphaTConditionalFlowMatcher):
    """AlphaT + Lipman et al. 2023 style target OT conditional flow matching. This class inherits the
    ConditionalFlowMatcher and override the compute_mu_t, compute_sigma_t and
    compute_conditional_flow functions in order to compute [2]'s flow matching.

    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path tx1, see (Eq.20) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t x1, 1 - (1 - sigma) t), see (Eq.20) [2].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma 1 - (1 - sigma) t

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t), see Eq.(21) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t)

        References
        ----------
        [1] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)