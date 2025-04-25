import torch
import torch.nn as nn
from smt import smt_t
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def conv7x7_bn_gelu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=(7,1),padding=(3,0), bias=False),
        nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=(1,7),padding=(0,3), bias=False),
        nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.GELU(),
    )



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.rgb = smt_t()

        self.fm_1 = FM(64,64)
        self.fm_2 = FM(128,64)
        self.fm_3 = FM(256,64)
        self.fm_4 = FM(512,64)
        self.fms = [self.fm_1,self.fm_2,self.fm_3,self.fm_4]
        self.dec = Decode(32,32,32,32)



    def forward(self, x):


        fuses = []
        B = x.shape[0]
        for i in range(self.rgb.num_stages):
            patch_embed = getattr(self.rgb, f"patch_embed{i + 1}")
            block = getattr(self.rgb, f"block{i + 1}")
            norm = getattr(self.rgb, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x,fuse = self.fms[i](x)
            fuses.append(fuse)

        pred1,pred2,pred3,pred4 = self.dec(fuses[0], fuses[1], fuses[2], fuses[3], 384)

        return pred1,pred2,pred3,pred4

    def load_pre(self, pre_model):

        self.rgb.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")


class FM(nn.Module):
    def __init__(self, dim,oup):
        super(FM, self).__init__()
        self.dim = oup
        self.conv_n = Attention(dim, self.dim)
        self.emb = Embed(self.dim,self.dim)
        self.x_conv = Attention(self.dim, dim)

    def forward(self,x):
        out = self.conv_n(x)
        out = self.emb(out)
        out_x = self.x_conv(out) + x
        return out_x,out

class Attention(nn.Module):
    def __init__(self, dim,oup):
        super(Attention, self).__init__()
        self.dim = dim//4
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim)
        self.ca_conv = nn.Conv2d(dim, oup, kernel_size=1)
        self.sa_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.sa_conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.sa_conv3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.sa_conv4 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        self.sa_fusion = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.conv_end = conv1x1_bn_relu(dim*2,oup)
    def forward(self, x):
        ca = self.ca(x)
        x = x * ca
        x1,x2,x3,x4 = torch.chunk(x,4,dim=1)
        sa = self.sa(x1)
        sa1 = self.sa_conv1(sa)
        sa2 = self.sa_conv2(sa1)
        sa3 = self.sa_conv3(sa2)
        sa4 = self.sa_conv4(sa3)

        x1 = sa1 * x1
        x2 = sa2 * x2
        x3 = sa3 * x3
        x4 = sa4 * x4
        ca_fusion = self.ca_conv(ca)
        sa_fusion = self.sa_fusion(torch.cat((sa1,sa2,sa3,sa4),1))
        out = self.conv_end(torch.cat((x,x1, x2, x3, x4),1))
        out = out * sa_fusion * ca_fusion

        return out

class Embed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(Embed, self).__init__()

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = residual + x
        return out

class MEF(nn.Module):
    def __init__(self, in1,in2,in3=None):
        super(MEF, self).__init__()
        if in3!=None:
            self.conv = BasicConv2d(in1 + in2 + in3,in1,1)
            self.dw = DWConv(in1)
        else:
            self.conv = BasicConv2d(in1 + in2, in1,1)
            self.dw = DWConv(in1)

    def forward(self, in1, in2=None, in3=None):
        if in3!=None:
            in2 = F.interpolate(in2, size=in1.size()[2:],mode='bilinear')
            in3 = F.interpolate(in3, size=in1.size()[2:], mode='bilinear')
            x = torch.cat((in1, in2, in3), 1)
            out = self.conv(x)
            out = self.dw(out)
        else:
            in2 = F.interpolate(in2, size=in1.size()[2:],mode='bilinear')
            x = torch.cat((in1, in2), 1)
            out = self.conv(x)
            out = self.dw(out)

        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LN(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        out = x_norm.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class Decode(nn.Module):
    def __init__(self, in1,in2,in3,in4):
        super(Decode, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c_up4 = nn.Sequential(
            nn.Conv2d(in4,in3,kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(in3),
            self.upsample2
        )
        self.c_up3 = nn.Sequential(
            nn.Conv2d(in3, in2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in2),
            self.upsample2
        )
        self.c_up2 = nn.Sequential(
            nn.Conv2d(in2, in1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in1),
            self.upsample2
        )
        self.c_up1 = nn.Sequential(
            nn.Conv2d(in1, in1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in1),
            self.upsample2
        )

        self.upb_4 = Block(in3)
        self.upb_3 = Block(in2)
        self.upb_2 = Block(in1)
        self.upb_1 = Block(in1)


        self.p_1 = nn.Sequential(
            nn.Conv2d(in_channels=in1, out_channels=in1//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in1//2),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=in1//2, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

        self.p2 = nn.Conv2d(in1, 1, kernel_size=3, padding=1)
        self.p3 = nn.Conv2d(in2, 1, kernel_size=3, padding=1)
        self.p4 = nn.Conv2d(in3, 1, kernel_size=3, padding=1)

        self.mf3 = MEF(in3,in4)
        self.mf2 = MEF(in2,in3,in4)
        self.mf1 = MEF(in1,in2,in3)

    def forward(self,x1,x2,x3,x4,s):

        x1_1, x1_2 = torch.chunk(x1, 2, 1)
        x2_1, x2_2 = torch.chunk(x2, 2, 1)
        x3_1, x3_2 = torch.chunk(x3, 2, 1)
        x4_1, x4_2 = torch.chunk(x4, 2, 1)

        x1 = x1_1 + x1_2
        x2 = x2_1 + x2_2
        x3 = x3_1 + x3_2
        x4 = x4_1 + x4_2

        r3 = self.mf3(x3, x4)
        r2 = self.mf2(x2,x3,x4)
        r1 = self.mf1(x1,x2,x3)

        up4 = self.upb_4(self.c_up4(x4))
        up3 = self.upb_3(self.c_up3(up4+r3))
        up2 = self.upb_2(self.c_up2(up3+r2))
        up1 = self.upb_1(self.c_up1(up2+r1))

        pred1 = self.p_1(up1)
        pred2 = F.interpolate(self.p2(up2), size=s, mode='bilinear')
        pred3 = F.interpolate(self.p3(up3), size=s, mode='bilinear')
        pred4 = F.interpolate(self.p4(up4), size=s, mode='bilinear')

        return pred1,pred2,pred3,pred4




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        mip = max(8, in_planes // ratio)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = self.sigmoid(max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim//2
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()
        self.sa_conv = nn.Conv2d(1,1,kernel_size=1)
        self.dw = DWBlock(self.dim)
        self.msa = KTM(self.dim)
        self.conv_end = nn.Conv2d(dim*2,dim,kernel_size=1)

    def forward(self, x):
        ca = self.ca(x)
        x = x * ca
        sa = self.sa_conv(self.sa(x))
        x = x * sa
        res = x
        x1,x2= torch.split(x,[self.dim,self.dim],1)
        x1 = self.dw(x1)
        x12 = self.msa(x1,x2)
        x = torch.cat((x,x12), 1)
        out = self.conv_end(x) + res
        return out


class DWConv(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LN(dim)
        self.res = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv_end = conv1x1_bn_relu(dim*2,dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.res(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.conv_end(torch.cat((shortcut,x),1))
        return x


class GRN(nn.Module):


    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DWBlock(nn.Module):

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class KTM(nn.Module):
    def __init__(self,channel=32):
        super(KTM, self).__init__()

        self.query_conv = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.key_conv = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.qk_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1,bias=False)
        self.value_conv_1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.value_conv_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)


        self.softmax = nn.Softmax(dim=1)

        self.conv_1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_2 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_out = BasicConv2d(channel*2, channel*2, 3, padding=1)


    def forward(self, x1, x2): # V

        m_batchsize, C, height, width = x1.size()
        proj_query = self.query_conv(x1)
        proj_key = self.key_conv(x2)
        energy = proj_query+proj_key
        energy = self.qk_conv(energy)
        energy = self.softmax(energy)

        proj_value_1 = self.value_conv_1(x1)
        proj_value_2 = self.value_conv_2(x2)

        out_1 = proj_value_1*energy
        out_1 = self.conv_1(out_1 + x1)

        out_2 = proj_value_2*energy
        out_2 = self.conv_2(out_2 + x2)
        x_out = self.conv_out(torch.cat((out_1,out_2),1))
        return x_out


if __name__ == '__main__':
    import torch
    import torchvision
    from thop import profile

    model = MyNet().cuda()

    a = torch.randn(1, 3, 384, 384).cuda()
    b = torch.randn(1, 3, 384, 384).cuda()
    flops, params = profile(model, (a,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
    # net = CatNet()
    # a = torch.randn([1, 3, 384, 384])
    # b = torch.randn([1, 3, 384, 384])
    #
    # s, s1, s2, s3 = net(a, b)
    # print("s.shape:", s.shape)
