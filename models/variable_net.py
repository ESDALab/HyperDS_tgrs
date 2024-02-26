import torch
import torch.nn  as nn
import sys
sys.path.append('/mnt/petrelfs/liuzili/code/OBDS')
from utils.positional_encoding import SineCosPE
import torch.nn.functional as F

class ResMLP(nn.Module):
    def __init__(self,in_channels):
        super(ResMLP, self).__init__()
        self.fc=nn.Sequential(nn.Linear(in_channels,in_channels),
                              # nn.Dropout(p=0.5),
                              nn.ReLU(inplace=True),
                              nn.Linear(in_channels, in_channels),
                              # nn.BatchNorm1d(in_channels)
                              )
    def forward(self,x):
        out=self.fc(x)
        return out+x

class VariableNet(nn.Module):
    '''
    inputs: b x token_num x in_channels (d_model)
    '''
    def __init__(self,token_num,in_channels,hidden_channels):
        super(VariableNet, self).__init__()
        self.in_channels=in_channels
        self.hidden_channels=hidden_channels
        self.token_num=token_num
        self.feat_fc_pre = nn.Linear(token_num, hidden_channels)
        self.feat_fc1 = nn.Linear(in_channels, hidden_channels+1)
        
        # self.data_fc=ResMLP(hidden_channels)
        self.cat_fc1=ResMLP(hidden_channels*1)
        # self.cat_fc2 = ResMLP(hidden_channels * 1)
        self.out_fc = nn.Linear(hidden_channels*1,1)
        self.pe=SineCosPE(1,N_freqs=hidden_channels//2//1,include_input=False)
        
        self.relu=nn.ReLU(inplace=True)

    def forward(self,meta_out:torch.Tensor,coord,coord_data):
        
        batch_size,dim,_,_=meta_out.shape
        meta_out = meta_out.view(batch_size,dim,-1)
        feat1 = self.feat_fc_pre(meta_out)   # b, 64, 128
        x = torch.bmm(coord, feat1)
        
        feat2 = self.feat_fc1(feat1.permute(0,2,1))  #b,128,128
        w1 = feat2[:,:,0:self.hidden_channels]  # b,128,128
        b1 = feat2[:,:,self.hidden_channels].unsqueeze(1)    # b,1,128
        
        x = torch.bmm(x, w1) + b1
        x=self.relu.forward(x)
        coord_data = coord_data.unsqueeze(2)
        coord_data_pe = self.pe(coord_data)
        cat_x = x + coord_data_pe
        x = self.cat_fc1(cat_x)
        x=x+cat_x
        x=self.out_fc.forward(x)
        return x