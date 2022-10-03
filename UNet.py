# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# October 2022
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int):
        super(UNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels : int = 1, out_channels : int = 1):
        super(UNet, self).__init__()

        self.block1 = UNetBlock(in_channels, 32)

        self.pool = nn.MaxPool2d(2,2)

        self.block2 = UNetBlock(32, 64)
        self.block3 = UNetBlock(64, 128)
        self.block4 = UNetBlock(128, 256)

        # Bend
        self.block5 = UNetBlock(256, 512)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, output_padding=0)
        self.block6 = UNetBlock(2*256, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, output_padding=0)
        self.block7 = UNetBlock(2*128, 128)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, output_padding=0)
        self.block8 = UNetBlock(2*64, 64)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, output_padding=0)
        self.block9 = UNetBlock(2*32, 32)

        self.conv10 = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        if (x.shape[2] & 15) != 0 or (x.shape[3] & 15) != 0:
            raise RuntimeError("Width and height need to be divisible by 16.")

        side1 = self.block1(x)
        x = self.pool(side1)

        side2 = self.block2(x)
        x = self.pool(side2)

        side3 = self.block3(x)
        x = self.pool(side3)

        side4 = self.block4(x)
        x = self.pool(side4)

        # Bend
        x = self.block5(x)

            
        x = self.up6(x)
        x = torch.cat((side4, x), dim=1)
        x = self.block6(x)

        x = self.up7(x)
        x = torch.cat((side3, x), dim=1)
        x = self.block7(x)

        x = self.up8(x)
        x = torch.cat((side2, x), dim=1)
        x = self.block8(x)

        x = self.up9(x)
        x = torch.cat((side1, x), dim=1)
        x = self.block9(x)

        x = self.conv10(x)
    
        return x

