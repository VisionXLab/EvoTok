import numpy as np
import torch
from torch import nn
import torch.distributed as dist

def gather(data):
    if dist.is_initialized():
        all_data = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
        dist.all_gather(all_data, data)
        all_data = torch.cat(all_data, dim=0)
    else:
        all_data = data
    return all_data


class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5, vq_warmup=0):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema or self.restart_unused_codes:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

        self.vq_warmup = vq_warmup
        if self.vq_warmup > 0:
            self.register_buffer("step", torch.tensor(0, dtype=torch.long))
            print(f"Warming up for {vq_warmup} steps.")
        
    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)

        return embed_idxs, distances

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B -1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x    
    
    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        # print("update buffers.")
        n_embed, embed_dim = self.weight.shape[0]-1, self.weight.shape[-1]
        
        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)
        
        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0,
                              index=idxs.unsqueeze(0),
                              src=vectors.new_ones(1, n_vectors)
                              )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        if self.vq_warmup > 0 and self.step >= self.vq_warmup:
            self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)
        if self.vq_warmup > 0:
            self.step += 1

        if self.restart_unused_codes:
            vectors = gather(vectors)
            
            if vectors.shape[0] < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            
            _vectors_random = vectors[torch.randperm(vectors.shape[0], device=vectors.device)][:n_embed]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)
            
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1-usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1-usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):
        # print("update embeddings.")
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        ).clamp(min=1.0)   # lots of zeros make the training unstable!!! -- clamp to 1.0
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs, distances = self.find_nearest_embedding(inputs)
        
        if self.ema and self.training:
            self._update_buffers(inputs, embed_idxs)
        
        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs, distances

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


class SharedResidualQuantizer(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.

    Arguments:
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(self,
                 n_embed,
                 embed_dim,
                 code_depth=(1,8),
                 decay=0.99,
                 show_usage=True,
                 restart_unused_codes=True,
                 commitment_loss='cumsum',
                 vq_warmup=0
                 ):
        super().__init__()
        
        self.code_depth = code_depth
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        self.decay = decay

        codebooks = [VQEmbedding(self.n_embed, 
                                 embed_dim, 
                                 decay=self.decay, 
                                 restart_unused_codes=restart_unused_codes,
                                 vq_warmup=vq_warmup
                                 ) for idx in range(max(self.code_depth))]
        self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss
        self.show_usage = show_usage
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(131072, max(self.code_depth))))

        print(f"Codebook size={n_embed}, dim={embed_dim}")
        print(f"Using SharedResidualQuantizer, vqgan_depth={self.code_depth[0]}, vqkd_depth={self.code_depth[1]}")
        print(f"restart_unused_codes={restart_unused_codes}")

    def reset_codebook_usage(self):
        self.codebook_used[:, :] = 0
        print("Reset codebook usage index.")


    def to_code_shape(self, x):
        x = torch.einsum('b c h w -> b h w c', x).contiguous()
        return x

    def to_latent_shape(self, x):
        x = torch.einsum('b h w c -> b c h w', x)
        return x

    def quantize(self, x):
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The code is selected by the residuals between x and quantized features by the previous codebooks.

        Arguments:
            x (Tensor): bottleneck feature maps to quantize.

        Returns:
            quant_list (list): list of sequentially aggregated and quantized feature maps by codebooks.
            codes (LongTensor): codewords index, corresponding to quants.

        Shape:
            - x: (B, h, w, embed_dim)
            - quant_list[i]: (B, h, w, embed_dim)
            - codes: (B, h, w, d)
        """
        B, h, w, embed_dim = x.shape

        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        d_norm_list = []
        aggregated_quants = torch.zeros_like(x)
        book_i = 0
        for i in range(max(self.code_depth)):
            quant, code, distances = self.codebooks[i](residual_feature)
            d_norm = torch.mean(torch.sum(distances**2, dim=-1))

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))
            d_norm_list.append(d_norm)
        
        codes = torch.cat(code_list, dim=-1)
        
        return quant_list, codes, d_norm_list

    def forward(self, x):
        x_reshaped = self.to_code_shape(x)
        quant_list, codes, d_norms = self.quantize(x_reshaped)

        commit_loss = self.compute_commitment_loss(x_reshaped, quant_list)
        vqkd_zq = self.to_latent_shape(quant_list[self.code_depth[1]-1])
        vqkd_zq = x + (vqkd_zq - x).detach()

        vqgan_zq = self.to_latent_shape(quant_list[self.code_depth[0]-1])
        vqgan_zq = x + (vqgan_zq - x).detach()

        perplexity = None
        min_encodings = None
        vq_loss = 0.0
        entropy_loss = 0.0
        codebook_usage = (0.0, 0.0)
        vqkd_d_norm, vqgan_d_norm = d_norms[self.code_depth[1]-1], d_norms[self.code_depth[0]-1]

        if self.show_usage:
            usage_codes = codes
            usage_codes = usage_codes.view(-1, usage_codes.shape[-1])
            cur_len = usage_codes.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = usage_codes[-cur_len:]
            codebook_usage = []
            for i in range(max(self.code_depth)):
                codebook_usage.append(len(torch.unique(self.codebook_used[..., i])) / self.n_embed)


        return (vqgan_zq, vqkd_zq), (vq_loss, commit_loss, entropy_loss, codebook_usage, vqkd_d_norm, vqgan_d_norm), (perplexity, min_encodings, codes)


    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []
        
        for idx, quant in enumerate(quant_list):
            partial_loss = (x-quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)
        
        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss
    
    @torch.no_grad()
    def embed_code(self, code):
        raise NotImplementedError   # TODO
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        
        embeds = torch.cat(embeds, dim=-2).sum(-2)
        embeds = self.to_latent_shape(embeds)

        return embeds
    
    @torch.no_grad()
    def embed_code_with_depth(self, code, to_latent_shape=False):
        assert (code.shape[-1] == self.code_depth[0]) or (code.shape[-1] == self.code_depth[-1])
        
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        embeds = []
        for i, code_slice in enumerate(code_slices):
            embeds.append(self.codebooks[i].embed(code_slice))

        embeds = torch.cat(embeds, dim=-2)
        
        return embeds, None
