import torch 
import torch.nn as nn
import torch.functional as F
from Config import *

# Define model 
class SiamNet(nn.Module):
    def __init__(self):
        super(SiamNet,self).__init__()

        # architecture (AlexNet)
        self.embedding_function = nn.Sequential(
            nn.Conv2d(3,96,11,2),    # conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,1, groups=2), # 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,3,1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,3,1, groups=2)
        )

        self.filter = nn.Conv2d(1,1,1,1)
        self._initialize_weight()

        self.config = Config()
    
    def forword(self,z,x):
        """
        forward computation:
        z: examplare, B*C*H*W
        x: search region, B*C*H1*W1 
        """
        z_embedding = self.embedding_function(z)
        x_embedding = self.embedding_function(x)

        # correlation of z and x
        xcorr_z = self.xcorr(z_embedding, x_embedding)

        score = self.adjust(xcorr_z)

        return score
    
    def xcorr(self,z,x):
        """
        correlation layer is conv z by x 
        """

        batch_size_x, channel_x, w_x, h_x = x.shape
        x = torch.reshape(x, (1, batch_size_x * channel_x, w_x, h_x))

        # group convolution
        out = F.conv2d(x, z, groups = batch_size_x)
        # Notice batch_size_out = 1, channel_out = "batch_size"
        batch_size_out, channel_out, w_out, h_out = out.shape
        
        xcorr_out = torch.reshape(out, (channel_out, batch_size_out, w_out, h_out))

        score = self.filter(xcorr_out)

        return score

    def _initialize_weight(self):
        """
        initialize network parameters by each layers.
        """
        tmp_layer_idx = 0
        for m in self.modules():
            # isinstance(object, classinfo) 
            # To determine whether an object is a known type
            if isinstance(m, nn.Conv2d):
                tmp_layer_idx +=1 
                if tmp_layer_idx < 6:
                    # kaiming initialization
                    nn.init.kaiming_normal_(m.weight.data, 
                                    mode='fan_out', nonlinearity='relu')
                else:
                    m.weight.data.fill_(1e-3)
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_loss(self, prediciton, label, weight):
        """
        weighted cross entropy loss
        """

        return F.binary_cross_entropy_with_logits(prediciton,
                                                label,
                                                weight,
                                                size_average=False) / self.config.batch_size


# here, questions about xcorr? how to implement conv x by z using group?