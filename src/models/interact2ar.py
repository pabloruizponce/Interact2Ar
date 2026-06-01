import torch
import torch.nn as nn
from tqdm import tqdm

from diffusion.cfg import ClassifierFreeSampleModel
from diffusion.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from diffusion.resample import create_named_schedule_sampler
from diffusion.respace import SpacedDiffusion, space_timesteps
from models.part_aware_interaction_denoiser import PartAwareInteractionDenoiser
from models.utils import set_requires_grad
from utils.dataset import MotionNormalizerTorch

PART_AWARE_DENOISER_TYPE = "part_aware_interaction"


class AutoRegressiveSampler:
    """Autoregressive sampler that rolls the diffusion model through time."""
    def __init__(
        self,
        sample_fn,
        pred_length,
        prefix_length,
        max_long_term=None,
        downsampling_long_term=None,
        required_frames=196,
    ):
        """Store memory-window sizes and the diffusion sampling function.

        Args:
            sample_fn: Diffusion sampling loop used for each autoregressive chunk.
            pred_length: Number of new frames predicted per chunk.
            prefix_length: Number of short-term context frames.
            max_long_term: Number of downsampled long-term memory tokens.
            downsampling_long_term: Frame stride used to build long-term memory.
            required_frames: Number of frames to generate when lengths are not provided.
        """
        print("Using the Interact2Ar autoregressive sampler")
        self.sample_fn = sample_fn
        self.pred_length = pred_length
        self.prefix_length = prefix_length
        self.max_long_term = max_long_term
        self.downsampling_long_term = downsampling_long_term
        self.required_frames = required_frames
        self.has_long_term = max_long_term is not None

    def sample(self, model, shape, cond, first_frame=None, lengths=None):
        """
        Generates samples with optional variable lengths and initial frame/motion.

        Args:
            model: The generative model.
            shape (tuple): The shape of the output tensor for a single iteration (B, T, D).
            cond: The conditioning tensor for the model.
            first_frame (torch.Tensor, optional): Can be either:
                - 2D tensor of shape (B, D): Single initial frame
                - 3D tensor of shape (B, T, D): Initial motion sequence
            lengths (torch.Tensor or list, optional): Desired output length for each batch item.
                                                      If None, uses self.required_frames for all.
        """
        # Initialize sampling parameters
        (
            lengths,
            max_len,
            n_iterations,
            full_motion,
            full_memory_length,
            start_idx,
        ) = self._initialize_sampling(shape, cond, lengths, first_frame)

        # Create initial prefix and mask
        cur_prefix, mask = self._create_initial_prefix_and_mask(
            shape, cond.device, first_frame, full_motion, full_memory_length, start_idx
        )

        # Main sampling loop
        print(f"Starting autoregressive sampling for {n_iterations} iterations")
        for i in tqdm(range(n_iterations)):
            # Generate sample
            sample = self.sample_fn(
                model=model,
                shape=shape,
                clip_denoised=False,
                progress=True,
                model_kwargs={"mask": mask, "cond": cond},
                prefix=cur_prefix,
            )

            # Store the newly predicted part
            prediction_start = start_idx + (i * self.pred_length)
            prediction_end = prediction_start + self.pred_length
            full_motion[:, prediction_start:prediction_end, :] = sample.clone()[
                :, -self.pred_length :, :
            ]

            # The absolute index after the latest prediction
            prediction_idx = prediction_end

            # Update prefix and mask for next iteration
            if prediction_idx >= full_memory_length:
                cur_prefix, mask = self._update_prefix_memory_full(
                    shape,
                    full_motion,
                    full_memory_length,
                    prediction_idx,
                    lengths,
                    cond.device,
                    start_idx,
                )
            else:
                cur_prefix, mask = self._update_prefix_memory_filling(
                    shape,
                    full_motion,
                    full_memory_length,
                    prediction_idx,
                    lengths,
                    cond.device,
                    start_idx,
                )

        # Return only the newly generated frames (after initial motion if provided).
        return full_motion[:, start_idx : start_idx + max_len, :]

    def _initialize_sampling(self, shape, cond, lengths, first_frame):
        """Allocate the full output buffer and resolve generation lengths.

        Args:
            shape: Per-chunk sampling shape ``(B, T, D)``.
            cond: Text-conditioning tensor whose device anchors new tensors.
            lengths: Optional desired output lengths for each batch item.
            first_frame: Optional seed frame or seed motion.

        Returns:
            tuple: Length tensor, maximum length, number of chunks, output buffer,
            full memory length before downsampling, and starting index.
        """
        # Handle lengths parameter
        if lengths is None:
            lengths = torch.full((shape[0],), self.required_frames, device=cond.device)
        elif isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=cond.device)

        max_len = torch.max(lengths).item()

        # Calculate the number of iterations we need to do
        if max_len % self.pred_length == 0:
            n_iterations = max_len // self.pred_length
        else:
            n_iterations = (max_len // self.pred_length) + 1

        # Determine if first_frame is a single frame (2D) or initial motion (3D)
        if first_frame is not None and first_frame.dim() == 3:
            # Initial motion provided (3D tensor)
            start_idx = first_frame.shape[1]
            total_frames = start_idx + (n_iterations * self.pred_length)

            # Create tensor to hold both initial and new motion
            full_motion = torch.zeros(shape[0], total_frames, shape[2]).to(cond.device)
            full_motion[:, :start_idx, :] = first_frame.clone().to(cond.device)
        else:
            # No initial motion, start from scratch
            start_idx = 0
            full_motion = torch.zeros(
                shape[0], n_iterations * self.pred_length, shape[2]
            ).to(cond.device)

        # Calculate full memory length (without downsampling)
        if self.has_long_term:
            full_memory_length = (
                self.downsampling_long_term * self.max_long_term
            ) + self.prefix_length
        else:
            full_memory_length = self.prefix_length

        return (
            lengths,
            max_len,
            n_iterations,
            full_motion,
            full_memory_length,
            start_idx,
        )

    def _create_initial_prefix_and_mask(
        self, shape, device, first_frame, full_motion, full_memory_length, start_idx
    ):
        """Create the initial prefix and mask for the first iteration."""
        # Case 1: Initial motion provided (3D tensor)
        if first_frame is not None and first_frame.dim() == 3:
            # Extract context from the end of initial motion
            context = full_motion[
                :, max(0, start_idx - full_memory_length) : start_idx, :
            ]
            context_len = context.shape[1]

            # Pad the context if it's shorter than the required memory length
            cur_prefix = torch.zeros(shape[0], full_memory_length, shape[2]).to(device)
            cur_prefix[:, -context_len:, :] = context

            # Create the initial mask for the padded prefix
            mask = torch.zeros(shape[0], full_memory_length + self.pred_length, 2).to(
                device
            )
            mask[:, (full_memory_length - context_len) : -self.pred_length, :] = 1
            mask[:, -self.pred_length :, :] = 1  # The predicted part is always unmasked

            # Apply long-term memory downsampling if needed
            if self.has_long_term:
                prefix_part_long_term = cur_prefix[:, : -self.prefix_length, :]
                prefix_part_short_term = cur_prefix[:, -self.prefix_length :, :]
                cur_prefix = torch.cat(
                    [
                        prefix_part_long_term[:, :: self.downsampling_long_term][
                            :, : self.max_long_term, :
                        ],
                        prefix_part_short_term,
                    ],
                    dim=1,
                )

                mask_part_long_term = mask[
                    :, : -self.prefix_length - self.pred_length, :
                ]
                mask_part_short_term = mask[
                    :, -self.prefix_length - self.pred_length :, :
                ]
                mask = torch.cat(
                    [
                        mask_part_long_term[:, :: self.downsampling_long_term][
                            :, : self.max_long_term, :
                        ],
                        mask_part_short_term,
                    ],
                    dim=1,
                )

            return cur_prefix, mask

        # Case 2: Single first frame (2D) or no first frame
        if self.has_long_term:
            cur_prefix = torch.zeros(
                shape[0], self.max_long_term + self.prefix_length, shape[2]
            ).to(device)
            mask = torch.ones(shape[0], shape[1], 2).to(device)
            # If we have a first_frame (2D), only mask out prefix_length - 1
            if first_frame is not None:
                cur_prefix[:, -1, :] = first_frame
                mask[:, : self.max_long_term + self.prefix_length - 1, :] = 0
            else:
                mask[:, : self.max_long_term + self.prefix_length, :] = 0
        else:
            cur_prefix = torch.zeros(shape[0], self.prefix_length, shape[2]).to(device)
            mask = torch.ones(shape[0], shape[1], 2).to(device)
            # If we have a first_frame (2D), only mask out prefix_length - 1
            if first_frame is not None:
                cur_prefix[:, -1, :] = first_frame
                mask[:, : self.prefix_length - 1, :] = 0
            else:
                mask[:, : self.prefix_length, :] = 0

        return cur_prefix, mask

    def _update_prefix_memory_full(
        self,
        shape,
        full_motion,
        full_memory_length,
        prediction_idx,
        lengths,
        device,
        start_idx,
    ):
        """Update prefix and mask when we have enough predictions to fill the full memory."""
        # Create mask based on variable lengths
        mask = self._create_variable_length_mask(
            shape, lengths, prediction_idx, device, start_idx
        )

        # Get the new prefix
        cur_prefix = full_motion[
            :, prediction_idx - full_memory_length : prediction_idx, :
        ]

        # If we have long term memory, subsample the long term part
        if self.has_long_term:
            cur_prefix = torch.cat(
                [
                    cur_prefix[:, :: self.downsampling_long_term][
                        :, : self.max_long_term, :
                    ],
                    cur_prefix[:, -self.prefix_length :],
                ],
                dim=1,
            )

        return cur_prefix, mask

    def _update_prefix_memory_filling(
        self,
        shape,
        full_motion,
        full_memory_length,
        prediction_idx,
        lengths,
        device,
        start_idx,
    ):
        """Update prefix and mask when still filling the memory."""
        # Create a zero memory
        cur_prefix = torch.zeros(
            shape[0],
            full_memory_length,
            shape[2],
        ).to(device)

        # Fill the memory with what we have predicted so far
        cur_prefix[:, -prediction_idx:, :] = full_motion[:, :prediction_idx, :]

        # If we have long term memory, subsample the long term part
        if self.has_long_term:
            cur_prefix = torch.cat(
                [
                    cur_prefix[:, :: self.downsampling_long_term][
                        :, : self.max_long_term, :
                    ],
                    cur_prefix[:, -self.prefix_length :],
                ],
                dim=1,
            )

        # Create the mask to not attend the padded memory
        mask = torch.zeros(shape[0], full_memory_length + self.pred_length, 2).to(
            device
        )

        # Unpad the part that has been already predicted
        mask[:, (full_memory_length - prediction_idx) :, :] = 1

        # Apply variable length masking on top
        # Note: prediction_idx is absolute, so we need to account for start_idx
        relative_idx = prediction_idx - start_idx
        start_idx_relative = relative_idx
        end_idx_relative = relative_idx + self.pred_length

        for b in range(shape[0]):
            if lengths[b] <= start_idx_relative:
                # This sequence is already complete
                mask[b, :, :] = 0
            elif lengths[b] < end_idx_relative:
                # This sequence will complete during this iteration
                frames_needed = lengths[b] - start_idx_relative
                frames_to_mask = self.pred_length - frames_needed
                # Mask from the end of the prediction region
                mask[b, -(frames_to_mask):, :] = 0

        # If we have long term memory, subsample the mask
        if self.has_long_term:
            mask = torch.cat(
                [
                    mask[:, :: self.downsampling_long_term][:, : self.max_long_term, :],
                    mask[:, -(self.prefix_length + self.pred_length) :],
                ],
                dim=1,
            )

        return cur_prefix, mask

    def _create_variable_length_mask(
        self, shape, lengths, prediction_idx, device, start_idx
    ):
        """
        Creates a mask for variable-length sequences after the memory filling phase.
        """
        # Convert absolute prediction_idx to relative (from start of generation)
        relative_idx = prediction_idx - start_idx
        start_idx_relative = relative_idx
        end_idx_relative = relative_idx + self.pred_length

        # Check if all sequences need the same treatment
        all_continuing = torch.all(lengths >= end_idx_relative)
        if all_continuing:
            # All sequences need more frames, no masking needed
            return None

        # Create per-batch masks
        mask = torch.ones(shape[0], shape[1], 2).to(device)

        for b in range(shape[0]):
            if lengths[b] <= start_idx_relative:
                # This sequence is already complete, mask everything
                mask[b, :, :] = 0
            elif lengths[b] < end_idx_relative:
                # This sequence will complete during this iteration
                frames_needed = lengths[b] - start_idx_relative
                frames_to_mask = self.pred_length - frames_needed
                # Mask the extra frames from the end
                mask[b, -frames_to_mask:, :] = 0

        return mask


class Interact2ArModel(nn.Module):
    """Interact2Ar autoregressive diffusion model for text-to-interaction generation.

    This is the top-level paper model: it owns CLIP text conditioning, the
    autoregressive memory schedule, diffusion training losses, and sampling.
    The part-aware denoiser is kept as an internal component so checkpoints keep
    the same ``model.denoiser.*`` state-dict structure.
    """

    def __init__(self, opt_diffusion, opt_denoiser):
        """
        Initializes the diffusion model with the specified options.

        Args:
            opt_diffusion: Configuration options for the diffusion process.
            opt_denoiser: Configuration options for the denoiser model.
        """
        super().__init__()

        self.opt = opt_diffusion  # Diffusion options
        self.text_embedder = getattr(self.opt, "TEXT_EMBEDDER", "clip").lower()
        if self.text_embedder != "clip":
            raise ValueError("Only CLIP text conditioning is supported in this release.")
        self.text_embed_dim = getattr(
            self.opt, "TEXT_EMB_DIM", getattr(self.opt, "CLIP_D_MODEL", 768)
        )

        # Check if the NORMALIZE option exists
        if not hasattr(opt_diffusion, "NORMALIZE"):
            self.opt.NORMALIZE = False

        if not hasattr(opt_diffusion, "USE_DDIM"):
            self.opt.USE_DDIM = False

        self.has_long_term = hasattr(opt_diffusion, "MAX_LONG_TERM")
        print("Using long term memory") if self.has_long_term else print(
            "No long term memory"
        )

        if opt_denoiser.TYPE != PART_AWARE_DENOISER_TYPE:
            raise ValueError(
                "This public release only supports the main-paper "
                f"{PART_AWARE_DENOISER_TYPE} denoiser."
            )

        print("Using PartAwareInteractionDenoiser")
        self.denoiser = PartAwareInteractionDenoiser(
            opt_denoiser, pred_length=self.opt.PRED_MOTION_LENGTH
        )

        # Diffusion setup for training
        self.betas = get_named_beta_schedule(
            self.opt.BETA_SCHEDULER, self.opt.DIFFUSION_STEPS
        )

        # In the case of having long term memory, we have to increase the prefix length
        # With that all the previous diffusion code can be used without any change
        self.diffusion_train = SpacedDiffusion(
            use_timesteps=space_timesteps(
                self.opt.DIFFUSION_STEPS, [self.opt.DIFFUSION_STEPS]
            ),
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
            lambda_losses=self.opt.LAMBDA_LOSSES
            if hasattr(self.opt, "LAMBDA_LOSSES")
            else None,
            autoregressive=True,
            prefix_length=self.opt.PREFIX_MOTION_LENGTH + self.opt.MAX_LONG_TERM
            if self.has_long_term
            else self.opt.PREFIX_MOTION_LENGTH,
        )

        self.sampler = create_named_schedule_sampler(
            self.opt.SAMPLER, self.diffusion_train
        )

        import clip

        # CLIP is the only text encoder used by the main-paper models.
        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        # Keep CLIP frozen so training reproduces the paper setup.
        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        # This lightweight transformer is the trainable text adapter.
        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.text_embed_dim,
            nhead=self.opt.CLIP_N_HEADS,
            dim_feedforward=self.opt.CLIP_DIM_FEEDFORWARD,
            dropout=self.opt.CLIP_DROPOUT,
            activation=self.opt.CLIP_ACTIVATION,
            batch_first=True,
        )
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer, num_layers=self.opt.CLIP_N_LAYERS
        )
        self.clip_ln = nn.LayerNorm(self.text_embed_dim)

        # Loaded lazily because the released paper model uses NORMALIZE=False.
        self.normalizer = None

    def text_process(self, caption, text_tensor=None):
        """
        Encodes raw text from a batch into a condition tensor.

        Args:
            caption (list of str): List of captions to encode.
        Returns:
            torch.Tensor: Condition tensor of shape (batch_size, d_model).
        """

        device = next(self.parameters()).device
        raw_text = caption

        import clip

        # Token IDs select the EOS embedding even when CLIP features are cached.
        with torch.no_grad():
            text_tokens = clip.tokenize(raw_text, truncate=True).to(device)

        if text_tensor is not None:
            clip_out = text_tensor.to(device)
        else:
            with torch.no_grad():
                # CLIP's transformer is sequence-first internally.
                x = self.token_embedding(text_tokens).type(self.dtype)
                x = x + self.positional_embedding.type(self.dtype)
                x = x.permute(1, 0, 2)
                x = self.clip_transformer(x)
                x = x.permute(1, 0, 2)
                clip_out = self.ln_final(x).type(self.dtype)

        # Adapt frozen CLIP features to the denoiser conditioning space.
        out = self.clipTransEncoder(clip_out)
        out = self.clip_ln(out)

        # CLIP marks EOS with the highest token id in each sequence.
        cond = out[torch.arange(out.shape[0]), text_tokens.argmax(dim=-1)]
        return cond

    def mask_cond(self, cond, cond_mask_prob=0.1, force_mask=False):
        """
        Applies classifier-free guidance masking to the condition tensor.

        Args:
            cond (torch.Tensor): Condition tensor of shape (batch_size, d_model).
            cond_mask_prob (float): Probability of masking the condition.
            force_mask (bool): If True, forces masking regardless of probability.
        Returns:
            torch.Tensor: Masked condition tensor.
            torch.Tensor: Mask indicating which parts were masked (optional).
        """
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.0:
            # 1 -> use null_cond, 0 -> use real cond
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * cond_mask_prob
            ).view([bs] + [1] * (len(cond.shape) - 1))
            return cond * (1.0 - mask), (1.0 - mask)
        else:
            return cond, None

    def generate_src_mask(self, T, length, init_idx):
        """
        Generates a source mask based on motion lengths.
        Args:
            T (int): Total time steps in the motion sequence.
            length (torch.Tensor): Tensor containing lengths of each motion in the batch.
        Returns:
            torch.Tensor: Source mask of shape (batch_size, T, 2).
        """
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                if init_idx[i] > 0:
                    src_mask[i, : init_idx[i], p] = 0
                # In the case we have long term memory, we have to add the max long term for properly masking
                if self.has_long_term:
                    src_mask[
                        i,
                        self.opt.MAX_LONG_TERM
                        + self.opt.PREFIX_MOTION_LENGTH
                        + length[i] :,
                        p,
                    ] = 0
                else:
                    src_mask[i, self.opt.PREFIX_MOTION_LENGTH + length[i] :, p] = 0
        return src_mask

    def compute_loss(self, batch):
        """
        Processes text, computes condition, and calculates the diffusion loss.

        Args:
            batch (tuple): A tuple containing:
                - word_embeddings (torch.Tensor): Word embeddings of shape (batch_size, seq_len, embedding_dim).
                - pos_one_hots (torch.Tensor): Positional one-hot encodings.
                - caption (list of str): List of captions for the batch.
                - sent_len (torch.Tensor): Lengths of the sentences.
                - motion (torch.Tensor): Motion data of shape (batch_size, T, input_feats).
                - m_length (torch.Tensor): Lengths of the motions.
                - _ (any): Placeholder for additional data, not used here.
        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed loss value. The one which is going to be optimized.
                - losses (dict): Dictionary containing detailed loss information.
        """
        (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            _,
            text_tensor,
            joints,
            init_idx,
        ) = batch

        # Process text to get condition
        cond = self.text_process(caption, text_tensor=text_tensor)
        x_start = motion
        B, T = x_start.shape[:2]

        # Apply classifier-free guidance masking
        cond, cond_mask = self.mask_cond(cond, 0.1)

        # Create motion mask
        seq_mask = self.generate_src_mask(T, m_length, init_idx).to(x_start.device)

        # Sample timesteps
        t, _ = self.sampler.sample(B, x_start.device)

        # Calculate diffusion loss
        losses = self.diffusion_train.training_losses(
            model=self.denoiser,
            x_start=x_start,
            t=t,
            model_kwargs={"mask": seq_mask, "cond": cond},
            loss_config=self.opt.LOSS_CONFIG,
            normalize=self.opt.NORMALIZE,
            t_bar=self.opt.T_BAR,
            target_joints=joints,
            init_idx=init_idx,
        )
        return losses["loss"].mean(), losses

    def generate_motion(self, batch, first_frame=None):
        """
        Generates motion from text for inference.

        Args:
            batch (tuple): A tuple the whole batch but for inference we only need:
                - caption (list of str): List of captions for the batch.
                - m_length (torch.Tensor): Lengths of the motions.
        Returns:
            torch.Tensor: Generated motion of shape (batch_size, T, input_feats).
        """
        (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            _,
            text_tensor,
            joints,
        ) = batch

        # Process text to get condition
        cond = self.text_process(caption, text_tensor=text_tensor)
        B = cond.shape[0]
        T = m_length[0]  # Assumes uniform length in batch for generation

        if self.opt.USE_DDIM:
            use_timesteps = space_timesteps(self.opt.DIFFUSION_STEPS, self.opt.STRATEGY)
        else:
            use_timesteps = space_timesteps(
                self.opt.DIFFUSION_STEPS, [self.opt.DIFFUSION_STEPS]
            )

        # Diffusion setup for inference; DDIM remains available for checkpoint compatibility.
        diffusion_test = SpacedDiffusion(
            use_timesteps=use_timesteps,
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
            autoregressive=True,
            prefix_length=self.opt.PREFIX_MOTION_LENGTH + self.opt.MAX_LONG_TERM
            if self.has_long_term
            else self.opt.PREFIX_MOTION_LENGTH,
        )

        # Wrap denoiser for classifier-free guidance
        cfg_model = ClassifierFreeSampleModel(self.denoiser, self.opt.CFG_WEIGHT)

        # Sampling function
        if self.opt.USE_DDIM:
            sample_fn = diffusion_test.ddim_sample_loop
        else:
            sample_fn = diffusion_test.p_sample_loop

        sampler = AutoRegressiveSampler(
            sample_fn,
            pred_length=self.opt.PRED_MOTION_LENGTH,
            prefix_length=self.opt.PREFIX_MOTION_LENGTH,
            max_long_term=self.opt.MAX_LONG_TERM if self.has_long_term else None,
            downsampling_long_term=self.opt.DONWSAMPLING_LONG_TERM
            if self.has_long_term
            else None,
            required_frames=T,
        )

        # The paper evaluation can seed sampling with the ground-truth first frame.
        first_frame_value = first_frame

        # Generate motion via autoregressive sampling loop
        output = sampler.sample(
            model=cfg_model,
            shape=(
                B,
                self.opt.MAX_LONG_TERM
                + self.opt.PREFIX_MOTION_LENGTH
                + self.opt.PRED_MOTION_LENGTH
                if self.has_long_term
                else self.opt.PRED_MOTION_LENGTH + self.opt.PREFIX_MOTION_LENGTH,
                self.denoiser.opt.MOTION_REP,
            ),
            cond=cond,
            first_frame=first_frame_value,
            lengths=m_length,
        )

        if self.opt.NORMALIZE:
            # Denormalize only for checkpoints trained with motion normalization.
            if self.normalizer is None:
                self.normalizer = MotionNormalizerTorch()
            print("Denormalizing output motion")
            output = self.normalizer.backward(output)

        return output

    def forward(self, batch, first_frame=None):
        """Generate motion for a batch.

        Args:
            batch: Batch tuple produced by the Inter-X dataloader or inference helper.
            first_frame: Optional first frame used to anchor autoregressive sampling.

        Returns:
            torch.Tensor: Generated motion sequence.
        """
        return self.generate_motion(batch, first_frame=first_frame)
