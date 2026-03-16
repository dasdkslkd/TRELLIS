"""
Two-Stage Sparse Structure VAE Trainer.

Supports three training modes controlled by the `training_mode` config:
  - "joint":  Train encoder + decoder_stage1 + decoder_stage2 together.
  - "stage1": Train encoder + decoder_stage1 only (ignore decoder_stage2).
  - "stage2": Freeze encoder + decoder_stage1, train decoder_stage2 only.

Loss design:
  Stage 1 (channels 0-7):
    - Channel 0 (occupancy):   Dice loss
    - Channels 1-7 (one-hot):  Cross-entropy loss (masked by occupancy)
  Stage 2 (channels 8-82):
    - 75 channels (25 pts × 3D coords): MSE loss (masked by occupancy)
  Regularization:
    - KL divergence on latent space
"""

from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer


class TwoStageSparseStructureVaeTrainer(BasicTrainer):
    """
    Trainer for Two-Stage Sparse Structure VAE.

    Config models should contain:
        - encoder:         SparseStructureEncoder   (in_channels = total channels)
        - decoder_stage1:  SparseStructureDecoder    (out_channels = stage1_channels)
        - decoder_stage2:  SparseStructureCondDecoder (out_channels = stage2_channels)

    Args:
        training_mode (str): "joint", "stage1", or "stage2".
        stage1_channels (int): Number of channels for stage 1 (default 8).
        stage2_channels (int): Number of channels for stage 2 (default 75).
        lambda_dice (float): Weight for dice loss (channel 0).
        occupancy_overlap_loss (str): Overlap loss for occupancy channel, "dice" or "tversky".
        tversky_alpha (float): False-positive weight used by Tversky loss.
        tversky_beta (float): False-negative weight used by Tversky loss.
        lambda_bce_occ (float): Weight for occupancy BCE loss (channel 0).
        occ_bce_pos_weight (float): Positive class weight for occupancy BCE.
        lambda_ce (float): Weight for cross-entropy loss (channels 1-7).
        lambda_mse (float): Weight for MSE loss (channels 8-82).
        lambda_kl (float): Weight for KL divergence.
        detach_stage1_for_stage2 (bool): Detach stage1 output before feeding to stage2.
        stage2_cond_source (str): "predicted" or "gt" - source of stage2 conditioning.
    """

    def __init__(
        self,
        *args,
        training_mode: str = 'joint',
        stage1_channels: int = 8,
        stage2_channels: int = 75,
        lambda_dice: float = 1.0,
        occupancy_overlap_loss: str = 'dice',
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
        lambda_bce_occ: float = 0.0,
        occ_bce_pos_weight: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_mse: float = 1.0,
        lambda_kl: float = 1e-3,
        detach_stage1_for_stage2: bool = False,
        stage2_cond_source: str = 'predicted',
        **kwargs
    ):
        self.training_mode = training_mode
        self.stage1_channels = stage1_channels
        self.stage2_channels = stage2_channels
        self.lambda_dice = lambda_dice
        self.occupancy_overlap_loss = occupancy_overlap_loss
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.lambda_bce_occ = lambda_bce_occ
        self.occ_bce_pos_weight = occ_bce_pos_weight
        self.lambda_ce = lambda_ce
        self.lambda_mse = lambda_mse
        self.lambda_kl = lambda_kl
        self.detach_stage1_for_stage2 = detach_stage1_for_stage2
        self.stage2_cond_source = stage2_cond_source

        super().__init__(*args, **kwargs)

    def init_models_and_more(self, **kwargs):
        """
        Override to freeze models based on training_mode before building optimizer.
        """
        if self.training_mode == 'stage2':
            # Freeze encoder and decoder_stage1
            for param in self.models['encoder'].parameters():
                param.requires_grad = False
            for param in self.models['decoder_stage1'].parameters():
                param.requires_grad = False
            if self.is_master:
                print(f'\n[TwoStage] Mode=stage2: Froze encoder and decoder_stage1.')
        elif self.training_mode == 'stage1':
            # Freeze decoder_stage2 if present
            if 'decoder_stage2' in self.models:
                for param in self.models['decoder_stage2'].parameters():
                    param.requires_grad = False
                if self.is_master:
                    print(f'\n[TwoStage] Mode=stage1: Froze decoder_stage2.')
        elif self.training_mode == 'joint':
            if self.is_master:
                print(f'\n[TwoStage] Mode=joint: All models trainable.')
        else:
            raise ValueError(f'Invalid training_mode: {self.training_mode}')

        super().init_models_and_more(**kwargs)

    def _activate_stage1(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply activations to stage 1 logits:
          - Channel 0: sigmoid (occupancy probability)
          - Channels 1-7: softmax (class probabilities)
        
        Returns:
            Activated tensor [B, stage1_channels, H, W, D]
        """
        occ = torch.sigmoid(logits[:, 0:1])                 # [B, 1, H, W, D]
        cls_probs = F.softmax(logits[:, 1:self.stage1_channels], dim=1)  # [B, 7, H, W, D]
        return torch.cat([occ, cls_probs], dim=1)

    def _compute_stage1_loss(self, logits: torch.Tensor, ss: torch.Tensor) -> dict:
        """
        Compute stage 1 losses.

        Args:
            logits: [B, stage1_channels, H, W, D] raw logits from decoder_stage1.
            ss: [B, C_total, H, W, D] ground truth.

        Returns:
            dict of loss terms.
        """
        terms = {}

        # --- Dice loss for channel 0 (occupancy) ---
        occ_logits = logits[:, 0:1]
        occ_pred = torch.sigmoid(occ_logits)
        occ_gt = ss[:, 0:1].float()
        intersection = (occ_pred * occ_gt).sum()
        if self.occupancy_overlap_loss == 'dice':
            dice_loss = 1.0 - (2.0 * intersection + 1.0) / (occ_pred.sum() + occ_gt.sum() + 1.0)
        elif self.occupancy_overlap_loss == 'tversky':
            false_pos = (occ_pred * (1.0 - occ_gt)).sum()
            false_neg = ((1.0 - occ_pred) * occ_gt).sum()
            dice_loss = 1.0 - (intersection + 1.0) / (
                intersection + self.tversky_alpha * false_pos + self.tversky_beta * false_neg + 1.0
            )
        else:
            raise ValueError(f'Invalid occupancy_overlap_loss: {self.occupancy_overlap_loss}')
        terms['dice'] = dice_loss
        pos_weight = torch.tensor(self.occ_bce_pos_weight, device=logits.device, dtype=logits.dtype)
        terms['bce_occ'] = F.binary_cross_entropy_with_logits(occ_logits, occ_gt, pos_weight=pos_weight)

        # --- Cross-entropy loss for channels 1-7 (one-hot categories) ---
        # Only compute on occupied voxels (where GT occ > 0.5)
        occ_mask = (occ_gt > 0.5).float()  # [B, 1, H, W, D]

        if occ_mask.sum() > 0:
            # logits[:, 1:8] are raw logits for 7 classes
            # ss[:, 1:8] are one-hot ground truth
            cls_logits = logits[:, 1:self.stage1_channels]    # [B, 7, H, W, D]
            cls_gt = ss[:, 1:self.stage1_channels].float()    # [B, 7, H, W, D]
            target_class = cls_gt.argmax(dim=1)                # [B, H, W, D]

            ce_loss = F.cross_entropy(cls_logits, target_class, reduction='none')  # [B, H, W, D]
            ce_loss = (ce_loss * occ_mask.squeeze(1)).sum() / occ_mask.sum().clamp(min=1)
            terms['ce'] = ce_loss
        else:
            terms['ce'] = torch.tensor(0.0, device=logits.device)

        return terms

    def _compute_stage2_loss(self, pred: torch.Tensor, ss: torch.Tensor) -> dict:
        """
        Compute stage 2 losses (MSE on sample point coordinates, masked by occupancy).

        Args:
            pred: [B, stage2_channels, H, W, D] raw output from decoder_stage2.
            ss: [B, C_total, H, W, D] ground truth.

        Returns:
            dict of loss terms.
        """
        terms = {}

        occ_gt = ss[:, 0:1].float()
        occ_mask = (occ_gt > 0.5).float()  # [B, 1, H, W, D]

        target = ss[:, self.stage1_channels:self.stage1_channels + self.stage2_channels].float()

        if occ_mask.sum() > 0:
            mse = F.mse_loss(pred, target, reduction='none')  # [B, stage2_channels, H, W, D]
            mse = (mse * occ_mask).sum() / (occ_mask.sum() * self.stage2_channels).clamp(min=1)
            terms['mse'] = mse
        else:
            terms['mse'] = torch.tensor(0.0, device=pred.device)

        return terms

    def training_losses(
        self,
        ss: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for the two-stage VAE.

        Args:
            ss: [B, C_total, H, W, D] tensor of voxel data.

        Returns:
            (loss_dict, status_dict)
        """
        # Encode
        z, mean, logvar = self.training_models['encoder'](
            ss.float(), sample_posterior=True, return_raw=True
        )

        terms = edict(loss=0.0)

        # KL divergence
        terms['kl'] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)

        # ==================== Stage 1 ====================
        if self.training_mode in ('joint', 'stage1'):
            stage1_logits = self.training_models['decoder_stage1'](z)
            s1_losses = self._compute_stage1_loss(stage1_logits, ss)
            terms['dice'] = s1_losses['dice']
            terms['bce_occ'] = s1_losses['bce_occ']
            terms['ce'] = s1_losses['ce']
            terms['loss'] = terms['loss'] + self.lambda_dice * terms['dice']
            terms['loss'] = terms['loss'] + self.lambda_bce_occ * terms['bce_occ']
            terms['loss'] = terms['loss'] + self.lambda_ce * terms['ce']
            terms['loss'] = terms['loss'] + self.lambda_kl * terms['kl']

        elif self.training_mode == 'stage2':
            # Stage 1 is frozen; run without gradients
            with torch.no_grad():
                stage1_logits = self.training_models['decoder_stage1'](z)
            terms['dice'] = torch.tensor(0.0, device=ss.device)
            terms['bce_occ'] = torch.tensor(0.0, device=ss.device)
            terms['ce'] = torch.tensor(0.0, device=ss.device)

        # ==================== Stage 2 ====================
        if self.training_mode in ('joint', 'stage2'):
            # Prepare condition for stage 2 decoder
            if self.stage2_cond_source == 'gt':
                # Use ground truth stage1 channels as condition
                stage1_cond = ss[:, :self.stage1_channels].float()
            elif self.stage2_cond_source == 'predicted':
                stage1_activated = self._activate_stage1(stage1_logits)
                if self.detach_stage1_for_stage2:
                    stage1_cond = stage1_activated.detach()
                else:
                    stage1_cond = stage1_activated
            else:
                raise ValueError(f'Invalid stage2_cond_source: {self.stage2_cond_source}')

            # For stage2 mode, z is from frozen encoder (already no_grad for encoder params,
            # but z itself needs grad for decoder_stage2 if not detaching)
            if self.training_mode == 'stage2':
                z_for_stage2 = z.detach()
            else:
                z_for_stage2 = z

            stage2_pred = self.training_models['decoder_stage2'](z_for_stage2, stage1_cond)
            s2_losses = self._compute_stage2_loss(stage2_pred, ss)
            terms['mse'] = s2_losses['mse']
            terms['loss'] = terms['loss'] + self.lambda_mse * terms['mse']

            # Add KL loss for stage2-only mode too
            if self.training_mode == 'stage2':
                # Don't add KL since encoder is frozen
                pass
        else:
            terms['mse'] = torch.tensor(0.0, device=ss.device)

        return terms, {}

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=1, verbose=False):
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        gts = []
        recons_stage1 = []
        recons_stage2 = []
        recons_full = []

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}

            z = self.models['encoder'](args['ss'].float(), sample_posterior=False)

            # Stage 1 reconstruction
            stage1_logits = self.models['decoder_stage1'](z)
            stage1_activated = self._activate_stage1(stage1_logits)
            stage1_recon = torch.zeros_like(stage1_logits)
            stage1_recon[:, 0:1] = (stage1_activated[:, 0:1] > 0.5).float()
            # One-hot from argmax
            cls_idx = stage1_activated[:, 1:self.stage1_channels].argmax(dim=1, keepdim=True)
            stage1_recon[:, 1:self.stage1_channels].scatter_(1, cls_idx, 1.0)

            # Stage 2 reconstruction
            if 'decoder_stage2' in self.models:
                stage2_pred = self.models['decoder_stage2'](z, stage1_activated)
                recons_stage2.append(stage2_pred)
                # Full reconstruction
                full_recon = torch.cat([stage1_recon, stage2_pred], dim=1)
                recons_full.append(full_recon)

            gts.append(args['ss'])
            recons_stage1.append(stage1_recon)

        sample_dict = {
            'gt': {'value': torch.cat(gts, dim=0), 'type': 'sample'},
            'recon_stage1': {'value': torch.cat(recons_stage1, dim=0), 'type': 'sample'},
        }
        if recons_stage2:
            sample_dict['recon_stage2'] = {'value': torch.cat(recons_stage2, dim=0), 'type': 'sample'}
        if recons_full:
            sample_dict['recon_full'] = {'value': torch.cat(recons_full, dim=0), 'type': 'sample'}
        return sample_dict

    @torch.no_grad()
    def miou(self, pred, target, thr=0.5):
        """
        Compute mIoU for one-hot categories on occupied voxels.
        pred: [B, 8, D, H, W]  (stage1 output)
        target: [B, 8, D, H, W]
        """
        occ = target[:, 0] > thr
        if occ.sum() == 0:
            return torch.tensor(0.0)
        match = pred[:, 1:self.stage1_channels].argmax(1) == target[:, 1:self.stage1_channels].argmax(1)
        return (match & occ).sum().float() / occ.sum().float().clamp(min=1)

    @torch.no_grad()
    def miou_occ(self, p, t, thr=0.5):
        """
        Compute occupancy IoU.
        """
        p, t = (p > thr).flatten(1), (t > thr).flatten(1)
        inter = (p & t).sum(1).float()
        union = (p | t).sum(1).float()
        return (inter / (union + 1e-8)).mean().item()

    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        if self.val_dataset is None:
            return {}
        for model in self.models.values():
            model.eval()

        dataloader = DataLoader(
            copy.deepcopy(self.val_dataset),
            batch_size=self.batch_size_per_gpu,
            shuffle=False,
            num_workers=0,
            collate_fn=self.val_dataset.collate_fn if hasattr(self.val_dataset, 'collate_fn') else None,
        )

        total_miou = 0.0
        total_miou_occ = 0.0
        total_mse = 0.0
        total_occ_precision = 0.0
        total_occ_recall = 0.0
        total_pred_occ_ratio = 0.0
        total_gt_occ_ratio = 0.0
        num_batches = 0

        for data in dataloader:
            args = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            ss = args['ss'].float()

            z = self.models['encoder'](ss, sample_posterior=False)

            # Stage 1
            stage1_logits = self.models['decoder_stage1'](z)
            stage1_activated = self._activate_stage1(stage1_logits)

            batch_miou = self.miou(stage1_activated, ss, thr=0.5)
            batch_miou_occ = self.miou_occ(stage1_activated[:, :1], ss[:, :1], thr=0.5)
            pred_occ = stage1_activated[:, :1] > 0.5
            gt_occ = ss[:, :1] > 0.5
            inter = (pred_occ & gt_occ).sum().float()
            pred_pos = pred_occ.sum().float()
            gt_pos = gt_occ.sum().float()
            total_miou += batch_miou
            total_miou_occ += batch_miou_occ
            total_occ_precision += (inter / pred_pos.clamp(min=1)).item()
            total_occ_recall += (inter / gt_pos.clamp(min=1)).item()
            total_pred_occ_ratio += pred_pos.div(pred_occ.numel()).item()
            total_gt_occ_ratio += gt_pos.div(gt_occ.numel()).item()

            # Stage 2
            if 'decoder_stage2' in self.models:
                stage2_pred = self.models['decoder_stage2'](z, stage1_activated)
                occ_mask = (ss[:, 0:1] > 0.5).float()
                target = ss[:, self.stage1_channels:self.stage1_channels + self.stage2_channels]
                if occ_mask.sum() > 0:
                    mse = ((stage2_pred - target) ** 2 * occ_mask).sum() / (occ_mask.sum() * self.stage2_channels)
                    total_mse += mse.item()

            num_batches += 1

        avg_miou = total_miou / max(num_batches, 1)
        avg_miou_occ = total_miou_occ / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_occ_precision = total_occ_precision / max(num_batches, 1)
        avg_occ_recall = total_occ_recall / max(num_batches, 1)
        avg_pred_occ_ratio = total_pred_occ_ratio / max(num_batches, 1)
        avg_gt_occ_ratio = total_gt_occ_ratio / max(num_batches, 1)

        if self.is_master:
            print(f'Validation mIoU: {avg_miou:.4f}')
            print(f'Validation mIoU_occ: {avg_miou_occ:.4f}')
            print(f'Validation occ_precision: {avg_occ_precision:.4f}')
            print(f'Validation occ_recall: {avg_occ_recall:.4f}')
            print(f'Validation pred_occ_ratio: {avg_pred_occ_ratio:.4f}')
            print(f'Validation gt_occ_ratio: {avg_gt_occ_ratio:.4f}')
            if 'decoder_stage2' in self.models:
                print(f'Validation MSE (stage2): {avg_mse:.6f}')

        for model in self.models.values():
            model.train()

        return {
            'miou': avg_miou,
            'miou_occ': avg_miou_occ,
            'occ_precision': avg_occ_precision,
            'occ_recall': avg_occ_recall,
            'pred_occ_ratio': avg_pred_occ_ratio,
            'gt_occ_ratio': avg_gt_occ_ratio,
            'mse_stage2': avg_mse,
        }
