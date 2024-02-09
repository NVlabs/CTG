"""
This file contains a collection of useful loss functions for use with torch tensors.
Partially borrowed from https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/loss_utils.py
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import (
    VEH_VEH_collision,
    VEH_PED_collision,
    PED_VEH_collision,
    PED_PED_collision,
)
import torch.nn.functional as F


def cosine_loss(preds, labels):
    """
    Cosine loss between two tensors.
    Args:
        preds (torch.Tensor): torch tensor
        labels (torch.Tensor): torch tensor
    Returns:
        loss (torch.Tensor): cosine loss
    """
    sim = torch.nn.CosineSimilarity(dim=len(preds.shape) - 1)(preds, labels)
    return -torch.mean(sim - 1.0)


def KLD_0_1_loss(mu, logvar):
    """
    KL divergence loss. Computes D_KL( N(mu, sigma) || N(0, 1) ). Note that
    this function averages across the batch dimension, but sums across dimension.
    Args:
        mu (torch.Tensor): mean tensor of shape (B, D)
        logvar (torch.Tensor): logvar tensor of shape (B, D)
    Returns:
        loss (torch.Tensor): KL divergence loss between the input gaussian distribution
            and N(0, 1)
    """
    return -0.5 * (1. + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def KLD_gaussian_loss(mu_1, logvar_1, mu_2, logvar_2):
    """
    KL divergence loss between two Gaussian distributions. This function
    computes the average loss across the batch.
    Args:
        mu_1 (torch.Tensor): first means tensor of shape (B, D)
        logvar_1 (torch.Tensor): first logvars tensor of shape (B, D)
        mu_2 (torch.Tensor): second means tensor of shape (B, D)
        logvar_2 (torch.Tensor): second logvars tensor of shape (B, D)
    Returns:
        loss (torch.Tensor): KL divergence loss between the two gaussian distributions
    """
    return -0.5 * (1. + \
                   logvar_1 - logvar_2 \
                   - ((mu_2 - mu_1).pow(2) / logvar_2.exp()) \
                   - (logvar_1.exp() / logvar_2.exp()) \
                   ).sum(dim=1).mean()


def KLD_discrete(logp,logq):
    """KL divergence loss between two discrete distributions. This function
    computes the average loss across the batch.

    Args:
        logp (torch.Tensor): log probability of first discrete distribution (B,D)
        logq (torch.Tensor): log probability of second discrete distribution (B,D)
    """
    return (torch.exp(logp)*(logp-logq)).sum(dim=1)


def log_normal(x, m, v, avails=None):
    """
    Log probability of tensor x under diagonal multivariate normal with
    mean m and variance v. The last dimension of the tensors is treated
    as the dimension of the Gaussian distribution - all other dimensions
    are treated as independent Gaussians. Adapted from CS 236 at Stanford.
    Args:
        x (torch.Tensor): tensor with shape (B, ..., D)
        m (torch.Tensor): means tensor with shape (B, ..., D) or (1, ..., D)
        v (torch.Tensor): variances tensor with shape (B, ..., D) or (1, ..., D)
        avails (torch.Tensor): availability of  x and m
    Returns:
        log_prob (torch.Tensor): log probabilities of shape (B, ...)
    """
    if avails is None:
        element_wise = -0.5 * (torch.log(v) + (x - m).pow(2) / v + np.log(2 * np.pi))
    else:
        element_wise = -0.5 * (torch.log(v) + ((x - m) * avails).pow(2) / v + np.log(2 * np.pi))
    log_prob = element_wise.sum(-1)
    return log_prob


def log_normal_mixture(x, m, v, w=None, log_w=None):
    """
    Log probability of tensor x under a uniform mixture of Gaussians.
    Adapted from CS 236 at Stanford.
    Args:
        x (torch.Tensor): tensor with shape (B, D)
        m (torch.Tensor): means tensor with shape (B, M, D) or (1, M, D), where
            M is number of mixture components
        v (torch.Tensor): variances tensor with shape (B, M, D) or (1, M, D) where
            M is number of mixture components
        w (torch.Tensor): weights tensor - if provided, should be
            shape (B, M) or (1, M)
        log_w (torch.Tensor): log-weights tensor - if provided, should be
            shape (B, M) or (1, M)
    Returns:
        log_prob (torch.Tensor): log probabilities of shape (B,)
    """

    # (B , D) -> (B , 1, D)
    x = x.unsqueeze(1)
    # (B, 1, D) -> (B, M, D) -> (B, M)
    log_prob = log_normal(x, m, v)
    if w is not None or log_w is not None:
        # this weights the log probabilities by the mixture weights so we have log(w_i * N(x | m_i, v_i))
        if w is not None:
            assert log_w is None
            log_w = torch.log(w)
        log_prob += log_w
        # then compute log sum_i exp [log(w_i * N(x | m_i, v_i))]
        # (B, M) -> (B,)
        log_prob = log_sum_exp(log_prob , dim=1)
    else:
        # (B, M) -> (B,)
        log_prob = log_mean_exp(log_prob , dim=1) # mean accounts for uniform weights
    return log_prob

def NLL_GMM_loss(x, m, v, pi, avails=None, detach=True, mode="sum"):
    """
    Log probability of tensor x under a uniform mixture of Gaussians.
    Adapted from CS 236 at Stanford.
    Args:
        x (torch.Tensor): tensor with shape (B, D)
        m (torch.Tensor): means tensor with shape (B, M, D) or (1, M, D), where
            M is number of mixture components
        v (torch.Tensor): variances tensor with shape (B, M, D) or (1, M, D) where
            M is number of mixture components
        logpi (torch.Tensor): log probability of the modes (B,M)
        detach (bool): option whether to detach all modes but the best one
        mode (string): mode of loss, sum or max

    Returns:
        -log_prob (torch.Tensor): log probabilities of shape (B,)
    """
    if v is None:
        v = torch.ones_like(m)

    # (B , D) -> (B , 1, D)
    x = x.unsqueeze(1)
    # (B, 1, D) -> (B, M, D) -> (B, M)
    if avails is not None:
        avails = avails.unsqueeze(1)
    log_prob = log_normal(x, m, v, avails=avails)
    if mode=="sum":
        if detach:
            max_flag = (log_prob==log_prob.max(dim=1,keepdim=True)[0])
            nonmax_flag = torch.logical_not(max_flag)
            log_prob_detach = log_prob.detach()
            NLL_loss = (-pi*log_prob*max_flag).sum(1).mean()+(-pi*log_prob_detach*nonmax_flag).sum(1).mean()
        else:
            NLL_loss = (-pi*log_prob).sum(1).mean()
    elif mode=="max":
        max_flag = (log_prob==log_prob.max(dim=1,keepdim=True)[0])
        NLL_loss = (-pi*log_prob*max_flag).sum(1).mean()
    return NLL_loss


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner.
    Adapted from CS 236 at Stanford.
    Args:
        x (torch.Tensor): a tensor
        dim (int): dimension along which mean is computed
    Returns:
        y (torch.Tensor): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner.
    Adapted from CS 236 at Stanford.
    Args:
        x (torch.Tensor): a tensor
        dim (int): dimension along which sum is computed
    Returns:
        y (torch.Tensor): log(sum(exp(x), dim))
    """

    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)

    return max_x + (new_x.exp().sum(dim)).log()


def project_values_onto_atoms(values, probabilities, atoms):
    """
    Project the categorical distribution given by @probabilities on the
    grid of values given by @values onto a grid of values given by @atoms.
    This is useful when computing a bellman backup where the backed up
    values from the original grid will not be in the original support,
    requiring L2 projection.
    Each value in @values has a corresponding probability in @probabilities -
    this probability mass is shifted to the closest neighboring grid points in
    @atoms in proportion. For example, if the value in question is 0.2, and the
    neighboring atoms are 0 and 1, then 0.8 of the probability weight goes to
    atom 0 and 0.2 of the probability weight will go to 1.
    Adapted from https://github.com/deepmind/acme/blob/master/acme/tf/losses/distributional.py#L42

    Args:
        values: value grid to project, of shape (batch_size, n_atoms)
        probabilities: probabilities for categorical distribution on @values, shape (batch_size, n_atoms)
        atoms: value grid to project onto, of shape (n_atoms,) or (1, n_atoms)
    Returns:
        new probability vectors that correspond to the L2 projection of the categorical distribution
        onto @atoms
    """

    # make sure @atoms is shape (n_atoms,)
    if len(atoms.shape) > 1:
        atoms = atoms.squeeze(0)

    # helper tensors from @atoms
    vmin, vmax = atoms[0], atoms[1]
    d_pos = torch.cat([atoms, vmin[None]], dim=0)[1:]
    d_neg = torch.cat([vmax[None], atoms], dim=0)[:-1]

    # ensure that @values grid is within the support of @atoms
    clipped_values = values.clamp(min=vmin, max=vmax)[:, None, :] # (batch_size, 1, n_atoms)
    clipped_atoms = atoms[None, :, None] # (1, n_atoms, 1)

    # distance between atom values in support
    d_pos = (d_pos - atoms)[None, :, None] # atoms[i + 1] - atoms[i], shape (1, n_atoms, 1)
    d_neg = (atoms - d_neg)[None, :, None] # atoms[i] - atoms[i - 1], shape (1, n_atoms, 1)

    # distances between all pairs of grid values
    deltas = clipped_values - clipped_atoms # (batch_size, n_atoms, n_atoms)

    # computes eqn (7) in distributional RL paper by doing the following - for each
    # output atom in @atoms, consider values that are close enough, and weight their
    # probability mass contribution by the normalized distance in [0, 1] given
    # by (1. - (z_j - z_i) / (delta_z)).
    d_sign = (deltas >= 0.).float()
    delta_hat = (d_sign * deltas / d_pos) - ((1. - d_sign) * deltas / d_neg)
    delta_hat = (1. - delta_hat).clamp(min=0., max=1.)
    probabilities = probabilities[:, None, :]
    return (delta_hat * probabilities).sum(dim=2)


def trajectory_loss(predictions, targets, availabilities, weights_scaling=None, crit=nn.MSELoss(reduction="none")):
    """
    Aggregated per-step loss between gt and predicted trajectories
    Args:
        predictions (torch.Tensor): predicted trajectory [B, (A), T, D]
        targets (torch.Tensor): target trajectory [B, (A), T, D]
        availabilities (torch.Tensor): [B, (A), T]
        weights_scaling (torch.Tensor): [D]
        crit (nn.Module): loss function

    Returns:
        loss (torch.Tensor)
    """
    assert availabilities.shape == predictions.shape[:-1]
    assert predictions.shape == targets.shape
    if weights_scaling is None:
        weights_scaling = torch.ones(targets.shape[-1]).to(targets.device)
    assert weights_scaling.shape[-1] == targets.shape[-1]
    target_weights = (availabilities.unsqueeze(-1) * weights_scaling)
    loss = torch.mean(crit(predictions, targets) * target_weights)
    return loss

def MultiModal_trajectory_loss(predictions, targets, availabilities, prob, weights_scaling=None, crit=nn.MSELoss(reduction="none"),calc_goal_reach=False):
    """
    Aggregated per-step loss between gt and predicted trajectories
    Args:
        predictions (torch.Tensor): predicted trajectory [B, M, (A), T, D]
        targets (torch.Tensor): target trajectory [B, (A), T, D]
        availabilities (torch.Tensor): [B, (A), T]
        prob (torch.Tensor): [B, M]
        weights_scaling (torch.Tensor): [D]
        crit (nn.Module): loss function

    Returns:
        loss (torch.Tensor)
    """

    if weights_scaling is None:
        weights_scaling = torch.ones(targets.shape[-1]).to(targets.device)
    assert weights_scaling.shape[-1] == targets.shape[-1]
    target_weights = (availabilities.unsqueeze(-1) * weights_scaling).unsqueeze(1)
    loss_v = crit(predictions,targets.unsqueeze(1))*target_weights
    if predictions.ndim==4:
        loss_v_aggregated = loss_v.sum(dim=[2,3])
    elif predictions.ndim==5:
        loss_v_aggregated = loss_v.sum(dim=[2,3,4])
    loss_v_detached = loss_v.detach()
    min_flag = (loss_v_aggregated==loss_v_aggregated.min(dim=1,keepdim=True)[0])
    nonmin_flag = torch.logical_not(min_flag)
    if predictions.ndim==4:
        min_weight = (min_flag*prob)[...,None,None]*target_weights
        nonmin_weight = (nonmin_flag*prob)[...,None,None]*target_weights
    elif predictions.ndim==5:
        min_weight = (min_flag*prob)[...,None,None,None]*target_weights
        nonmin_weight = (nonmin_flag*prob)[...,None,None,None]*target_weights
    loss = ((loss_v*min_weight).sum()+(loss_v_detached*nonmin_weight).sum())/availabilities.sum()
    if calc_goal_reach:
        last_inds = batch_utils().get_last_available_index(availabilities)  # [B, (A)]
        num_frames = availabilities.shape[-1]
        goal_mask = TensorUtils.to_one_hot(last_inds, num_class=num_frames)
        goal_mask = goal_mask.unsqueeze(1).unsqueeze(-1)
        goal_loss = ((loss_v*min_weight*goal_mask).sum()+(loss_v_detached*nonmin_weight*goal_mask).sum())/goal_mask.sum()
        return loss, goal_loss
    else:
        return loss

        

def goal_reaching_loss(predictions, targets, availabilities, weights_scaling=None, crit=nn.MSELoss(reduction="none")):
    """
    Final step loss between gt and predicted trajectories (normally used in conjunction with a forward dynamics model)
    Args:
        predictions (torch.Tensor): predicted trajectory [B, (A), T, D]
        targets (torch.Tensor): target trajectory [B, (A), T, D]
        availabilities (torch.Tensor): [B, (A), T]
        weights_scaling (torch.Tensor): [D]
        crit (nn.Module): loss function

    Returns:
        loss (torch.Tensor)
    """
    # compute loss mask by finding the last available target
    num_frames = availabilities.shape[-1]
    last_inds = batch_utils().get_last_available_index(availabilities)  # [B, (A)]
    goal_mask = TensorUtils.to_one_hot(last_inds, num_class=num_frames)  # [B, (A), T] with the last frame set to 1
    # filter out samples that do not have available frames
    available_samples_mask = availabilities.sum(-1) > 0  # [B, (A)]
    goal_mask = goal_mask * available_samples_mask.unsqueeze(-1).float()  # [B, (A), T]
    goal_loss = trajectory_loss(
        predictions,
        targets,
        availabilities=goal_mask,
        weights_scaling=weights_scaling,
        crit=crit
    )
    return goal_loss



def lane_regulation_loss(lane_flag,agent_mask):
    return (lane_flag.mean(-1)*agent_mask).sum()/agent_mask.sum()

def weighted_trajectory_loss(
    predictions,
    targets,
    target_weights,
    total_count,
    weights_scaling=None,
    crit=nn.MSELoss(reduction="none"),
):
    """
    Aggregated per-step loss between gt and predicted trajectories
    Args:
        predictions (torch.Tensor): predicted trajectory [B, (A), T, D]
        targets (torch.Tensor): target trajectory [B, (A), T, D]
        weights (torch.Tensor): [B, (A), T]
        total_count (float)
        weight_scaling (torch.Tensor): [D], Defaults to None.
        crit (nn.Module): loss function

    Returns:
        loss (torch.Tensor)
    """
    assert target_weights.shape == predictions.shape[:-1]
    assert predictions.shape == targets.shape
    if weights_scaling is None:
        weights_scaling = torch.ones(targets.shape[-1]).to(targets.device)
    assert weights_scaling.shape[-1] == targets.shape[-1]
    target_weights = target_weights.unsqueeze(-1) * weights_scaling
    loss = torch.sum(crit(predictions, targets) * target_weights) / total_count
    return loss


def weighted_multimodal_trajectory_loss(
    predictions,
    targets,
    target_weights,
    probability,
    total_count,
    weights_scaling=None,
    crit=nn.MSELoss(reduction="none"),
):
    """
    Aggregated per-step loss between gt and predicted trajectories
    Args:
        predictions (torch.Tensor): predicted trajectory [B, M, A, T, D]
        targets (torch.Tensor): target trajectory [B, A, T, D]
        target_weights (torch.Tensor): [B, A, T]
        probability (torch.Tensor): [B,M]
        total_count (float)
        weight_scaling (torch.Tensor): [D], Defaults to None.
        crit (nn.Module): loss function

    Returns:
        loss (torch.Tensor)
    """
    assert target_weights.shape == predictions.shape[:-1]
    assert predictions.shape == targets.shape
    if weights_scaling is None:
        weights_scaling = torch.ones(targets.shape[-1]).to(targets.device)
    assert weights_scaling.shape[-1] == targets.shape[-1]
    target_weights = target_weights.unsqueeze(-1) * weights_scaling
    err = (
        crit(
            targets.unsqueeze(1).repeat(1, predictions.size(1), 1, 1, 1),
            predictions,
        )
        * target_weights.unsqueeze(1)
        * probability[:, :, None, None, None]
    )
    max_idx = torch.max(probability, dim=-1)[1]
    max_mask = torch.zeros([*err.shape[:2], 1, 1, 1], dtype=torch.bool).to(err.device)
    max_mask[torch.arange(0, err.size(0)), max_idx] = True
    nonmax_mask = ~max_mask
    loss = (
        torch.sum((err * max_mask)) + torch.sum((err * nonmax_mask).detach())
    ) / total_count
    return loss


def likelihood_loss(likelihood):
    return 1.0 - torch.mean(likelihood)

def lane_regularization_loss(lane_flags, weights, total_count, probability=None):
    """penalizing the vehicle for exiting drivable area

    Args:
        lane_flags (torch.Tensor): 1 for in the lane, 0 for out of the lane, [B, (M), (A), T, 1]
        weights (torch.Tensor): [B, (A), T]
        total_count (float):
        probability (torch.Tensor, optional): [B,M]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if probability is None:
        loss = torch.sum(weights.unsqueeze(-1) * (1.0 - lane_flags)) / total_count
    else:
        if lane_flags.ndim == 4:
            probability = probability[:, :, None, None]
        elif lane_flags.ndim == 5:
            probability = probability[:, :, None, None, None]
        loss = (
            torch.sum(
                weights.unsqueeze(-1).unsqueeze(1) * (1.0 - lane_flags) * probability
            )
            / total_count
        )
        return loss
    return loss


def goal_reaching_loss(
    predictions,
    targets,
    availabilities,
    weights_scaling=None,
    crit=nn.MSELoss(reduction="none"),
):
    """
    Final step loss between gt and predicted trajectories (normally used in conjunction with a forward dynamics model)
    Args:
        predictions (torch.Tensor): predicted trajectory [B, (A), T, D]
        targets (torch.Tensor): target trajectory [B, (A), T, D]
        availabilities (torch.Tensor): [B, (A), T]
        weights_scaling (torch.Tensor): [D]
        crit (nn.Module): loss function

    Returns:
        loss (torch.Tensor)
    """
    # compute loss mask by finding the last available target
    num_frames = availabilities.shape[-1]
    last_inds = batch_utils().get_last_available_index(availabilities)  # [B, (A)]
    goal_mask = TensorUtils.to_one_hot(
        last_inds, num_class=num_frames
    )  # [B, (A), T] with the last frame set to 1
    # filter out samples that do not have available frames
    available_samples_mask = availabilities.sum(-1) > 0  # [B, (A)]
    goal_mask = goal_mask * available_samples_mask.unsqueeze(-1).float()  # [B, (A), T]
    goal_loss = trajectory_loss(
        predictions,
        targets,
        availabilities=goal_mask,
        weights_scaling=weights_scaling,
        crit=crit,
    )
    return goal_loss


def collision_loss(pred_edges: Dict[str, torch.Tensor], col_funcs=None):
    """
    Calculate collision loss among predicted edges along a batch of trajectories
    Args:
        pred_edges (dict): A dict that maps collision types to box locations
        col_funcs (dict): A dict of collision functions (implemented in tbsim.utils.geometric_utils)

    Returns:
        collision loss (torch.Tensor)
    """
    if col_funcs is None:
        col_funcs = {
            "VV": VEH_VEH_collision,
            "VP": VEH_PED_collision,
            "PV": PED_VEH_collision,
            "PP": PED_PED_collision,
        }

    coll_loss = 0.0
    for et, fun in col_funcs.items():
        if et not in pred_edges:
            continue
        edges = pred_edges[et]
        if edges.shape[0] == 0:
            continue
        dis = fun(
            edges[..., 0:3],
            edges[..., 3:6],
            edges[..., 6:8],
            edges[..., 8:],
        ).min(dim=-1)[
            0
        ]  # take min distance across time steps
        coll_loss += torch.mean(torch.sigmoid(-dis - 4.0))  # smooth collision loss
    return coll_loss

def collision_loss_masked(edges, type_mask, weight=None, col_funcs=None):
    if col_funcs is None:
        col_funcs = {
            "VV": VEH_VEH_collision,
            "VP": VEH_PED_collision,
            "PV": PED_VEH_collision,
            "PP": PED_PED_collision,
        }

    coll_loss = 0.0
    for k,v in type_mask.items():
        if edges.shape[0] == 0:
            continue
        dis = col_funcs[k](
            edges[..., 0:3],
            edges[..., 3:6],
            edges[..., 6:8],
            edges[..., 8:],
        ).min(dim=-1)[0]
        coll_loss_tensor = torch.sigmoid(-dis - 4.0)*v
        if weight is not None:
            coll_loss += torch.sum(coll_loss_tensor*weight)/((v*weight).sum()+1e-3)
        else:
            coll_loss += torch.sum(coll_loss_tensor)/(v.sum()+1e-3)
    return coll_loss


def discriminator_loss(likelihood_pred,likelihood_GT):
    label = torch.cat((torch.zeros_like(likelihood_pred),torch.ones_like(likelihood_GT)),0)
    return F.binary_cross_entropy(torch.cat((likelihood_pred,likelihood_GT)),label)

def compute_pred_loss(recon_loss_type,pred_batch,target_traj,avails,prob,weights_scaling=None):
    if "z" in pred_batch:
        z1 = torch.argmax(pred_batch["z"],dim=-1)
    else:
        z1 = None
    if recon_loss_type=="NLL":
        assert "logvar" in pred_batch["x_recons"]
        bs, M, T, D = pred_batch["trajectories"].shape
        var = (torch.exp(pred_batch["x_recons"]["logvar"])+torch.ones_like(pred_batch["x_recons"]["logvar"])*1e-4).reshape(bs,M,-1)
        if z1 is not None:
            var = torch.gather(var,1,z1.unsqueeze(-1).repeat(1,1,var.size(-1)))
        avails = avails.unsqueeze(-1).repeat(1, 1, target_traj.shape[-1]).reshape(bs, -1)
        pred_loss = NLL_GMM_loss(
            x=target_traj.reshape(bs,-1),
            m=pred_batch["trajectories"].reshape(bs,M,-1),
            v=var,
            pi=prob,
            avails=avails
        )
        pred_loss = pred_loss.mean()


    elif recon_loss_type=="MSE":
        
        pred_loss = MultiModal_trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=avails,
            prob = prob,
            weights_scaling=weights_scaling,
        )
    else:
        raise NotImplementedError("{} is not implemented".format(recon_loss_type))
    return pred_loss