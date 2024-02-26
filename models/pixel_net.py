import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelWiseFC(nn.Module):
    def __init__(self, inDim, outDim, ifRelu):
        super().__init__()
        self.inD = inDim
        self.outD = outDim
        self.ifRelu = ifRelu
        if ifRelu:
            self.relu = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, inX, paraT):
        bsize = paraT.size()[0]
        dimW = self.inD * self.outD
        Y = torch.matmul(paraT[:, 0:dimW].view(bsize, 1, self.outD, self.inD), inX) + \
            paraT[:, dimW:dimW+self.outD].unsqueeze(1).unsqueeze(-1)
        if self.ifRelu:
            Y = self.relu(Y)
        return Y

class PixelWiseAdpNet(nn.Module):
    def __init__(self, in_channel, out_channel, feat_channel, FCDims, feat_wh, out_wh, sample_num):
        super().__init__()
        
        self.feat_wh = feat_wh
        self.out_wh = out_wh
        self.out_ch = out_channel
        self.in_ch = in_channel
        self.FCDims = FCDims
        self.sample_num = sample_num
        self.pixelWFC = []
        din = in_channel
        c_total = 0
        for fd in FCDims:
            self.pixelWFC = self.pixelWFC + [PixelWiseFC(din, fd, True)]
            c_total += din * fd + fd
            din = fd
        c_total += din * self.out_ch + self.out_ch
        self.conv_coord_data = nn.Conv1d(out_channel,in_channel, 1, 1, 0)
        self.conv_feat = nn.Conv2d(feat_channel, c_total, 1, 1, 0)

        for i, graph in enumerate(self.pixelWFC):
            self.add_module('pixelWFC_{}'.format(i), graph)
        self.outWFC = PixelWiseFC(din, out_channel, False)
        # self.actF = nn.Sigmoid()

        
        
        
    def flattenX(self, x):
        if self.flag == 'grid':
            x = x.permute(0,2,3,4,1).contiguous()
            x = torch.flatten(x, 1, 3)
        elif self.flag == 'stn':
            x = x.permute(0,2,1).contiguous()
        return x
    
    def cal_HyperMPL(self, x, p_MPL, h, w):
        
        Bsize, point_num = x.shape[0], x.shape[2]
        x = self.flattenX(x).unsqueeze(-1)
        din = self.in_ch
        dim_beg = 0
        for fd, wFC in zip(self.FCDims, self.pixelWFC):
            dim_end = dim_beg + din * fd + fd
            x = wFC(x, p_MPL[:, dim_beg:dim_end])
            dim_beg = dim_end
            din = fd
        
        dim_end = dim_beg + din * self.out_ch + self.out_ch
        
        x = self.outWFC(x, p_MPL[:, dim_beg:dim_end]).squeeze(-1)

        # x = self.actF(x)
        if self.flag == 'grid':
            x = x.view(Bsize, self.sample_num, h, w, self.out_ch)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        elif self.flag == 'stn':
            x = x.view(Bsize, point_num, self.out_ch)
            x = x.permute(0, 2, 1).contiguous()
        return x
    
    def get_loc_idx(self, w2k_x, w2k_y, y_b, x_b, y_e, x_e):
        
        x_e_idx = torch.nonzero(w2k_x[0,:]<=x_e)[:,0]
        x_b_idx = torch.nonzero(w2k_x[0,:]>x_b)[:,0]
        x_idx = list(set(x_e_idx.tolist()) & set(x_b_idx.tolist()))

        y_e_idx = torch.nonzero(w2k_y[0,:]<=y_e)[:,0]
        y_b_idx = torch.nonzero(w2k_y[0,:]>y_b)[:,0]
        y_idx = list(set(y_e_idx.tolist()) & set(y_b_idx.tolist()))

        idx = list(set(x_idx) & set(y_idx))
        
        return idx    

    
    def forward(self, MLP_feature, coord_em, coord_data, flag, device, **kwargs):
        self.flag = flag
        if MLP_feature.shape[-2:] != self.feat_wh:
            MLP_feature = F.adaptive_avg_pool2d(MLP_feature, self.feat_wh)
        # import pdb
        # pdb.set_trace()
        coord_data = self.conv_coord_data(coord_data.permute(0,2,1).contiguous())
        inX = coord_em + coord_data
        
        feat = self.conv_feat(MLP_feature)
        del MLP_feature
        BSize, em_ch, point_num = inX.shape
        out_h, out_w = self.out_wh
        _, _, aH, aW = feat.size()
        psH, psW = out_h//aH, out_w//aW
        
        if flag == 'grid':
            inX = inX.view(BSize, em_ch, self.sample_num, out_h, out_w)
            
            out_grid = torch.zeros(BSize, self.out_ch, self.sample_num, out_h, out_w).to(device)
            for ky in range(aH):
                for kx in range(aW):
                    y_b = ky * psH
                    x_b = kx * psW
                    y_e = min(y_b + psH, out_h)
                    x_e = min(x_b + psW, out_w)
                    x = inX[:, :, :, y_b:y_e, x_b:x_e]
                    p_MLP = feat[:, :, ky, kx]
                    out_grid[:, :, :, y_b:y_e, x_b:x_e]  = self.cal_HyperMPL(x, p_MLP, y_e-y_b, x_e-x_b)
            return out_grid
        elif flag == 'stn':
            w2k_x = kwargs['w2k_x']/kwargs['dx']
            w2k_y = kwargs['w2k_y']/kwargs['dy']
            out_stn = torch.zeros(BSize, self.out_ch, point_num).to(device)

            for ky in range(aH):
                for kx in range(aW):
                    y_b = ky * psH
                    x_b = kx * psW
                    y_e = min(y_b + psH, out_h)
                    x_e = min(x_b + psW, out_w)
                    idx = self.get_loc_idx(w2k_x,w2k_y,y_b,x_b,y_e,x_e)
                    p_MLP = feat[:, :, ky, kx]
                    if len(idx) != 0:
                        out_stn[:,:,idx] = self.cal_HyperMPL(inX[:,:,idx], p_MLP, y_e-y_b, x_e-x_b)         
            return out_stn
        else:
            NotImplementedError

        
    
