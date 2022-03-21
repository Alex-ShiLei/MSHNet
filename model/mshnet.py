r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from .base.merge import merge
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation


class MsimilarityHyperrelationNetwork(nn.Module):
    def __init__(self, backbone, use_original_imgsize,shot=1):
        super(MsimilarityHyperrelationNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        self.shot=shot
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        #print(self.bottleneck_ids)#[0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2]
        #print(self.lids)#[1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
        #print(self.stack_ids)#[ 3,  9, 13]
        #print(self.feat_ids)#[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.backbone.eval()
        self.cross_entropy_loss =nn.CrossEntropyLoss()#(weight=torch.tensor([0.2,0.8]).cuda())
        self.merge=merge(shot,nsimlairy=list(reversed(nbottlenecks[-3:])),criter=self.cross_entropy_loss)
    def forward(self, query_img, support_img, support_mask,gt=None):
        sup_feats=[]#shot
        corrs=[]
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            for i in range(self.shot):
                support_feats = self.extract_feats(support_img[:,i,:,:,:], self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats = self.mask_feature(support_feats, support_mask[:,i,:,:])
                corr,sups = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)#[corr_l4, corr_l3, corr_l2]
                corrs.append(corr)#s,l,b,c,h,w
                sup_feats.append(sups)#s,l,n,b,2*c,hw

        logit_mask,loss = self.merge(sup_feats,corrs,gt)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[-2:], mode='bilinear', align_corners=True)

        return logit_mask,loss

    def mask_feature(self, features, support_mask):#bchw
        bs=features[0].shape[0]
        initSize=((features[0].shape[-1])*2,)*2
        support_mask = support_mask.unsqueeze(1).float()
        support_mask = F.interpolate(support_mask,initSize, mode='bilinear', align_corners=True)
        for idx, feature in enumerate(features):
            feat=[]
            if support_mask.shape[-1]!=feature.shape[-1]:
                support_mask = F.interpolate(support_mask, feature.size()[2:], mode='bilinear', align_corners=True)
            for i in range(bs):
                featI=feature[i].flatten(start_dim=1)#c,hw
                maskI=support_mask[i].flatten(start_dim=1)#hw
                featI = featI * maskI
                maskI=maskI.squeeze()
                meanVal=maskI[maskI>0].mean()
                realSupI=featI[:,maskI>=meanVal]
                if maskI.sum()==0:
                    realSupI=torch.zeros(featI.shape[0],1).cuda()
                feat.append(realSupI)#[b,]ch,w
            features[idx] = feat#nfeatures ,bs,ch,w
        return features

    def predict_mask_nshot(self, batch, nshot):
        logit_mask,loss = self(batch['query_img'], batch['support_imgs'], batch['support_masks'],batch['query_mask'])
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

        logit_mask_agg = logit_mask.argmax(dim=1)
        return logit_mask_agg

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
