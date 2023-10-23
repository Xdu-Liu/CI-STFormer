import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class TS_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(TS_TCN, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.avg_pool_t = nn.AdaptiveAvgPool2d(1)
        self.max_pool_t = nn.AdaptiveMaxPool2d(1)
        self.sigmoid_t = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        N,C,T,V=x.size()
        x0 = x.permute(0,2,1,3).contiguous() # N T C V
        Q = self.avg_pool_t(x0) # N T 1 1
        K = self.max_pool_t(x0) # N T 1 1
        Q = Q.squeeze(-1).squeeze(-1)
        K = K.squeeze(-1).squeeze(-1)
        atten = self.sigmoid_t(torch.einsum('nt,nm->ntm', (Q, K)))   # N T T  
        xs = self.bn(torch.einsum('nctv,ntm->ncmv', (x, atten)))      
        return xs
        
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 TX,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 1) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 1
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  
        ))

        #self.branches.append(nn.Sequential(
            #TS_TCN(in_channels, branch_channels, kernel_size=1, stride=stride),
            #nn.BatchNorm2d(branch_channels)  
        #))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class CI_SFormer(nn.Module):
    def __init__(self, in_channels, out_channels, A, PA, groups=8, coff_embedding=4, num_subset=3,t_stride=1,t_padding=0,t_dilation=1,bias=True,first=False,residual=True):
        super(CI_SFormer, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.out_channels = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.groups=groups
        self.pa = Variable(torch.from_numpy(np.reshape(PA.astype(np.float32),[3,1,6,6]).repeat(groups,axis=1)), requires_grad=False)
        self.Decouplepap = nn.Parameter(torch.tensor(np.reshape(PA.astype(np.float32),[3,1,6,6]), dtype=torch.float32, requires_grad=True).repeat(1,groups,1,1), requires_grad=True)
        self.A1 = Variable(torch.from_numpy(np.reshape(A.astype(np.float32),[3,1,25,25]).repeat(groups,axis=1)), requires_grad=False)
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32),[3,1,25,25]), dtype=torch.float32, requires_grad=True).repeat(1,groups,1,1), requires_grad=True)
        self.m = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        
        self.twice_1 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
        self.twice_2 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)        

        self.twice_11 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
        self.twice_21 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)   

        self.t_1 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
        self.t_2 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)  

        self.s_1 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
        self.s_2 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)  

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False) 
        self.sigmoid_1 = nn.Sigmoid()

        self.conv1 = nn.Conv2d(
            out_channels//8,
            (out_channels//8) * 3,
            kernel_size=(1, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * 3,
            kernel_size=(1, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        self.conv_s = nn.Conv2d(in_channels, out_channels, 1,bias=bias)
        self.conv_p = nn.Conv2d(in_channels, out_channels//8, 1,bias=bias)
        self.conv_e = nn.Conv2d(out_channels//8, out_channels, 1,bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        self.base_channel=64

    def forward(self, x):
        N,C,T,V = x.size()
        pa = self.pa.cuda(x.get_device())
        x1_pap = self.Decouplepap + pa
        norm_learn_pap = x1_pap.repeat(1,(self.out_channels//self.groups)//8,1,1)
        x1_final=torch.zeros([N,self.num_subset,self.out_channels//8,6,6],dtype=torch.float,device='cuda').detach()
        A1 = self.A1.cuda(x.get_device())
        A = A1 + self.DecoupleA
        norm_learn_A = A.repeat(1,self.out_channels//self.groups,1,1)
        A_final=torch.zeros([N,self.num_subset,self.out_channels,25,25],dtype=torch.float,device='cuda').detach()
        y = None
        y1 = None
        x0 = x
        N,C,T,V = x.size()
        x1 = self.conv_p(x)  # n 0-c//2 t [v]
        x2 = self.conv_s(x)  # n c//2-c t [v]
        x1_0 = x1[:,:,:,3].unsqueeze(-1)
        x1_1 = x1[:,:,:,20].unsqueeze(-1)
        x1_3 = x1[:,:,:,7].unsqueeze(-1)
        x1_6 = x1[:,:,:,11].unsqueeze(-1)
        x1_9 = x1[:,:,:,14].unsqueeze(-1)
        x1_12= x1[:,:,:,18].unsqueeze(-1)
        x1_part = torch.cat([x1_0,x1_1,x1_3,x1_6,x1_9,x1_12],dim=3) # N C//2 T 6
        A1 = x1_part.permute(0, 3, 1, 2).contiguous() # N 6 C//2 T
        A1 = self.m(A1)
        A1 = self.relu(self.t_1(A1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1))
        A2 = x1.permute(0, 3, 1, 2).contiguous() # N 6 C//2 T
        A2 = self.m(A2)
        A2 = self.relu(self.t_2(A2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1))
        A12 = self.soft(torch.einsum('nv,nw->nvw', (A1, A2))) # N 6 25
        x1_GCN = torch.einsum('nctv,nqv->nctq', (x1, A12)) # N C//2 T 6
        xs = x1_GCN
        for i in range(self.num_subset):        
            xm = self.avg_pool(xs.permute(0,3,1,2).contiguous()) # N V 1 1
            Q = self.relu(self.twice_1(xm.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1))
            K = self.relu(self.twice_2(xm.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1))
            atten = self.soft(torch.einsum('nv,nw->nvw', (Q, K)))  
            atten = atten.unsqueeze(1)
            atten = atten.repeat(1,self.out_channels//8,1,1)  # N C V V
            x1_final[:,i,:,:,:] = atten * 0.5  + norm_learn_pap[i]
        m = x1_GCN
        m = self.conv1(m)
        n, kc, t, v = m.size()
        m = m.view(n, self.num_subset, kc// self.num_subset, t, v)
        m = torch.einsum('nkctv,nkcvw->nctw', (m, x1_final))   
        y = m
        xs = x2
        for i in range(self.num_subset):        
            xm = self.avg_pool(xs.permute(0,3,1,2).contiguous()) # N V 1 1
            Q = self.relu(self.twice_11(xm.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1))
            K = self.relu(self.twice_21(xm.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1))
            atten = self.soft(torch.einsum('nv,nw->nvw', (Q, K)))  
            atten = atten.unsqueeze(1)
            atten = atten.repeat(1,self.out_channels,1,1)  # N C V V
            A_final[:,i,:,:,:] = atten * 0.5  + norm_learn_A[i]
        m = x2         
        m = self.conv2(m)
        n, kc, t, v = m.size()
        m = m.view(n, self.num_subset, kc// self.num_subset, t, v)
        m = torch.einsum('nkctv,nkcvw->nctw', (m, A_final)) 
        y1 = m
        A1l = y.permute(0, 3, 1, 2).contiguous() # N 6 C//2 T
        A1l = self.m(A1l)
        A1l = self.relu(self.s_1(A1l.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1))
        A2l = y1.permute(0, 3, 1, 2).contiguous() # N 6 C//2 T
        A2l = self.m(A2l)
        A2l = self.relu(self.s_2(A2l.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1))
        A12l = self.soft(torch.einsum('nv,nw->nvw', (A1l, A2l))) # N 6 25
        yl = torch.einsum('nctv,nvq->nctq', (y, A12l)) # N C//2 T 25                    
        yl = self.conv_e(yl)
        y = yl * 0.5 + y1
        m = y
        q = self.avg_pool(m)
        q = self.conv2_1(q.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) 
        q = self.sigmoid_1(q)
        q = m  *  q.expand_as(m) + m
        y = q
        y = self.bn(y)
        y += self.down(x0)
        y = self.relu(y)
        return y

class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
        
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, PA, TX, stride=1, residual=True, kernel_size=5, dilations=[1,2,3]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = CI_SFormer(in_channels, out_channels, A, PA)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, TX, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_6=None, graph_args=dict(), in_channels=3, drop_out=0):
        super(Model, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = self.graph.A # 3,25,25
        if graph_6 is None:
            raise ValueError()
        else:
            Graph = import_class(graph_6)
            self.graph_6 = Graph(**graph_args)
        PA = self.graph_6.A # 3,6,6
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * 64 * num_point)
        self.to_joint_embedding = nn.Linear(in_channels, 64)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 64))        
        self.A_vector = self.get_A(graph, 1)
        self.num_point = num_point
        base_channel = 64
        self.l1 = TCN_GCN_unit(base_channel, base_channel, A, PA, 64, residual=False)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, PA, 64)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, PA, 64)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, PA, 64)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, PA, 32,stride=2)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, PA, 32)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, PA, 32)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, PA, 16, stride=2)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, PA, 16)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, PA, 16)
        self.first_tram = nn.Sequential(
                nn.AvgPool2d((4,1)),
                nn.Conv2d(64, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        self.second_tram = nn.Sequential(
                nn.AvgPool2d((4,1)),
                nn.Conv2d(64, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        
        self.third_tram = nn.Sequential(
                nn.AvgPool2d((2,1)),
                nn.Conv2d(128, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))  
        
    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        p = self.A_vector
        p = torch.tensor(p,dtype=torch.float)
        x = p.to(x.device).expand(N*M*T, -1, -1) @ x
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        
        x = self.l1(x)
        x1=x
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x2=x
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x3=x
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        x1=self.first_tram(x1)   # x2 (N*M,64,75,25)
        x2=self.second_tram(x2)  # x2 (N*M,64,75,25)
        x3=self.third_tram(x3)   # x3(N*M,128,75,25)
        x=x+x2+x3+x1

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
