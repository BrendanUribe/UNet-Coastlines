import torch # py torch deep learning library
import torch.nn as nn # neural network tools
import torch.nn.functional as F # needed for F.interpolate (upsampling the multi-scale edge outputs)

# double convolution block for UNet
class DoubleConv(nn.Module): # recognize this part as a neural network (pytorch)
    def __init__(self, in_channels, out_channels): # setup funnction for the block
        # inputs: in_channels (number of input channels), out_channels (number of output channels)
        # rgb 3 channels in, 1 channel out for binary mask 

        super().__init__()

        # layers run in order 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # 2d conv layer to learn patterns (3x3 kernal, 1 padding to not shrink)
            nn.ReLU(inplace=True), # ReLU activation function 
            nn.Conv2d(out_channels, out_channels, 3, padding=1), # 2nd 2d layer
            nn.ReLU(inplace=True),
        )
        # 2 conv layers thus double conv block

    def forward(self, x): # move data forward 'x' is feature map
        return self.conv(x) # output processed features 


class UNet(nn.Module): # main unet model
    def __init__(self, in_channels=3, num_classes=3): # input channels (3 for rgb), output channels (3 for rgb), num_classes (3 for rgb)
        super().__init__() # initialize pytorch structure

        # Encoder - downsampling side that extracts features while reducing image size rbg input, 64 feature maps output, then 128, then 256
        # each encoder block helps learn more complex features 
        self.enc1 = DoubleConv(in_channels, 64) 
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2) # reduce image size by half to learn larger scale features 

        # Bottleneck - deepest part receiving 256 feature maps output 516 features maps, learns most cmpressed image info
        self.bottleneck = DoubleConv(256, 512)

        # Decoder - upsamples bottleneck to double image size then decode feature maps to reconstruct image, 512 to 256, then 256 to 128, then 128 to 64
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # second 'head' for learning land/water/cloud and coastlines
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1) # final output mask for segmentation

        self.edge_head_d3 = nn.Conv2d(256, 1, kernel_size=1) # coarse final output mask for edge detection at decoder level 3
        self.edge_head_d2 = nn.Conv2d(128, 1, kernel_size=1) # medium final output mask for edge detection at decoder level 2
        self.edge_head_d1 = nn.Conv2d(64, 1, kernel_size=1) # fine final output mask for edge detection at decoder level 1
        self.edge_fuse = nn.Conv2d(3, 1, kernel_size=1) # fuse the three edge outputs into a single edge output

        # Output - converts final 64 feature maps to desired... 1 masked image
        # each pixel is 1 predicted value (land/water)
        # self.final = nn.Conv2d(64, out_channels, kernel_size=1) *removed since now have 2 headed approach

    def forward(self, x):
        # Encoder - first second and third encoder blocks for downsampling 
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck - e passed thru bottlesneck b
        b = self.bottleneck(self.pool(e3))

        # Decoder - b is upsampled 
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1) # skip connection combining decoder features with matching encoder features 
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # return self.final(d1) # final output mask * removed since now have 2 headed approach
        seg_out = self.seg_head(d1) # segmentation output

        edge_d3 = self.edge_head_d3(d3)   # (B, 1, 64, 64) 
        edge_d2 = self.edge_head_d2(d2)   # (B, 1, 128, 128)
        edge_d1 = self.edge_head_d1(d1)   # (B, 1, 256, 256)

        target_size = edge_d1.shape[-2:]
        edge_d3_up = F.interpolate(edge_d3, size=target_size, mode='bilinear', align_corners=False)
        edge_d2_up = F.interpolate(edge_d2, size=target_size, mode='bilinear', align_corners=False)

        edge_fused = self.edge_fuse(torch.cat([edge_d3_up, edge_d2_up, edge_d1], dim=1)) # fuse the three edge outputs into a single edge output

        return seg_out, [edge_d3_up, edge_d2_up, edge_d1, edge_fused] # return both outputs