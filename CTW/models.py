import torch
from torch import nn
from helper import *
from GMM import GMM


# +
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        input_channels = 9,
        parse_channels = 17,
    ):
        super().__init__()

        self.encoder = GMM(channelA=4, channelB=4)
        self.parse_net = ParseUnet()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels + 4, init_dim, 7, padding = 3)
        
        self.semantic_mapper = nn.Conv2d(parse_channels, 4, 1, bias=False)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        context_dim = 512
        num_heads = 4

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            dim_head = dim_in // num_heads
            if ind <= 1:
                attn = None
#                 attn = Residual(PreNorm(dim_in, LinearAttention(dim_in)))
            else:
                attn = SpatialTransformer(
                            dim_in, num_heads, dim_head, context_dim=context_dim
                        )

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn,
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        
        mid_dim = dims[-1]
        dim_head = mid_dim // num_heads
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = SpatialTransformer(
                            mid_dim, num_heads, dim_head, context_dim=context_dim
                        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            dim_head = dim_out // num_heads
            if ind > 1:
#                 attn = Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                attn = None
            else:
                attn = SpatialTransformer(
                            dim_out, num_heads, dim_head, context_dim=context_dim
                        )

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn,
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, cond, context=None):
        
        cloth, cloth_mask, parse, pose = cond
        shape_cond = self.semantic_mapper(torch.cat((parse, pose), dim = 1))
        
        context = self.encoder(torch.cat((cloth, cloth_mask), dim = 1), shape_cond)

#         context = self.encoder(torch.cat((cloth, cloth_mask), dim = 1))
        x = torch.cat((x, shape_cond), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            if isinstance(attn, SpatialTransformer):
                x = attn(x, context)
            elif isinstance(attn, Residual):
                x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, context)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            if isinstance(attn, SpatialTransformer):
                x = attn(x, context)
            elif isinstance(attn, Residual):
                x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
    def get_parse(self, warp_cloth, cond):
        return self.parse_net(warp_cloth, cond)
        
    
class ParseUnet(nn.Module):
    def __init__(
        self,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        input_channels=9,
        parse_channels=14,
        resnet_block_groups = 8,
    ):
        super().__init__()

        init_dim = dim

        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)
        
        self.semantic_mapper = nn.Conv2d(parse_channels, 3, 1, bias=False)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(Block, groups = resnet_block_groups)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_dim = parse_channels

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, warp_cloth, cond):
        
        cloth, cloth_mask, parse, pose = cond
        
        parse = self.semantic_mapper(parse)

        x = torch.cat((parse, pose, warp_cloth), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x)
            h.append(x)
            
            x = block2(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)
        return self.final_conv(x)
