import math

import torch

from utils.util import make_anchors

# function to calculate appropriate padding
def pad(k, p=None, d=1):
    if d > 1:   # if dilatation involved
        k = d * (k - 1) + 1 # adjust the effective kernel size, accounting for extra spacing
    if p is None:   # if no padding provided
        p = k // 2  # half-padding
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv

# Convolutional block
class Conv(torch.nn.Module):
    
    # parameters: input channels, output channels, kernel size, stride, padding, dilation, groups
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False) # False to disable bias (redundant with BatchNorm)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)   # common scale and shift for YOLO models
        self.relu = torch.nn.SiLU(inplace=True) # Sigmoid Linear Unit

    # Conv2D -> BatchNorm2D -> SiLU
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    # optimization trick to faster the inference phase, 
    # we can mathematically combine BatchNorm parameters into Conv2d weights and biases
    def fuse_forward(self, x):
        return self.relu(self.conv(x))  # BatchNorm step is skipped


# Bottleneck block
class Residual(torch.nn.Module):
    # parameters: input channels, add indicates if the residual block adds a skip connection
    # the block is channel-preserving
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),   # here the bottleneck
                                         Conv(ch, ch, 3))   # two convolution with kernel size 3
        # note: this does not change the channel dimension in the bottleneck (it could be 0.5 * ch)

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


# C2f block (actually named C2f cause Cross Stage Partial is in v4, v5)
class CSP(torch.nn.Module):
    # parameters: input channels, output channels, number of residual blocks,
    # add indicates if each residual block adds a skip connection
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        
        # split the input channels into two branches, each with half the output channels
        self.conv1 = Conv(in_ch, out_ch // 2)   # first branch split, will not be transformed (a skip connection), will be concatenated later
        self.conv2 = Conv(in_ch, out_ch // 2)   # second branch split, will pass through n residual blocks
        # note: this performs two separate convs, not one followed by split (as in C2f block architecture)

        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))  # sequence of n residual blocks
        
        # final convolution to combine the outputs of the two branches and the residual blocks
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)  # 2 branches + n residual blocks
        
    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]  # first branch split and second branch split
        y.extend(m(y[-1]) for m in self.res_m)  # concatenate the outputs of the residual blocks to the list y
        return self.conv3(torch.cat(y, dim=1))  # final convolution


# Spatial Pyramid Pooling block (this is actually a SPPF block)
class SPP(torch.nn.Module):
    # parameters: input channels, output channels, kernel size
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)    # reduce the number of channels by half, to reduce computation before pooling
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)   # max pooling with kernel size k and stride 1, padding k // 2
        self.conv2 = Conv(in_ch * 2, out_ch)
        # note: in_ch // 2 from conv1 plus 3 more from the pooling steps, so 4 × (in_ch // 2) = in_ch * 2

    def forward(self, x):
        x = self.conv1(x)   # reduce the number of channels by half
        y1 = self.res_m(x)  # first pooling step
        y2 = self.res_m(y1) # second pooling step
        # the third polling step: y3 = self.res_m(y2) is done here below       
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))    # concatenate the outputs of the pooling steps and the original input, and pass through conv2


# Backbone network (DarkNet backbone)
class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]       # first conv layer, downsample by 2 (stride=2)
        p2 = [Conv(width[1], width[2], 3, 2),       # second conv layer, downsample by 2 (stride=2)
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)  # * unpacks the list into arguments
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


# Feature Pyramid Network (FPN) block - Neck
class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


# Distribution Focal Loss (DFL) module
class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


# Detect block (multi-scale detection head)
# outputs x, y, w, h, class_scores
class Head(torch.nn.Module):
    # placeholders to be set during inference with make_anchors
    anchors = torch.empty(0)
    strides = torch.empty(0)

    # parameters: number of classes, filters (number of channels in the last conv layer)
    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels, each coordinate (x, y, w, h) is predicted as a distribution over 16 bins
        self.nc = nc  # number of classes, 80 for COCO
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor = nc + ch * 4 (bounding box encoded by DFL)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)   # intermediate channel size used in the classification branch
        c2 = max((filters[0] // 4, self.ch * 4))    # intermediate channel size used in the bounding box regression branch

        self.dfl = DFL(self.ch)
        # classification branch: Conv -> Conv -> Conv2D
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3),
                                                           Conv(c1, c1, 3),
                                                           torch.nn.Conv2d(c1, self.nc, 1)) for x in filters)
        #  bounding box regression branch: Conv -> Conv -> Conv2D
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3),
                                                           Conv(c2, c2, 3),
                                                           torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)
        # note: 4 * self.ch because we have 4 coordinates (x, y, w, h) for each anchor box

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
            
        # training mode
        if self.training:
            return x
        
        # inference mode
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)) # computer anchors and strides

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)   # reshape per-feature outputs into unified format
        box, cls = x.split((self.ch * 4, self.nc), 1)   # split into box and class logits
        
        # DFL decoding
        a, b = torch.split(self.dfl(box), 2, 1) # left/right for center-based box
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)    # final bounding box and class confidence (after sigmoid)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth)    # backbone network
        self.fpn = DarkFPN(width, depth)    # feature pyramid network (neck)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


# YOLO v8 architecture variants

def yolo_v8_n(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes)


def yolo_v8_s(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_m(num_classes: int = 80):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes)


def yolo_v8_l(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_x(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth, num_classes)
