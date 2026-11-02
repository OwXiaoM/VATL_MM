"""Independent Stage-1 implementation inspired by VesselTok (CC BY-NC upstream).

Core: Fourier point embeddings -> FPS queries -> encoder cross/self attention ->
Gaussian tokens -> decoder self attention -> query cross attention -> occupancy.
No SIREN auto-decoder, analytic graph-distance decoder, or learned transform.
"""
import math
import torch
from torch import nn


@torch.no_grad()
def farthest_points(points,mask,count):
    if points.ndim!=3 or mask.shape!=points.shape[:2]: raise ValueError('Batched points and padding mask required')
    b,n,_=points.shape
    valid=~mask
    if not valid.any(1).all(): raise ValueError('Empty graphs are gated before encoding')
    centroid=(points*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
    distance=((points-centroid[:,None])**2).sum(-1).masked_fill(mask,-1)
    current=distance.argmax(1)
    nearest=torch.full((b,n),float('inf'),device=points.device)
    selected=[]; sampled_mask=[]; rows=torch.arange(b,device=points.device)
    for step in range(count):
        selected.append(current); sampled_mask.append(step>=valid.sum(1))
        candidate=points[rows,current]
        nearest=torch.minimum(nearest,((points-candidate[:,None])**2).sum(-1))
        nearest=nearest.masked_fill(mask,-1)
        current=nearest.argmax(1)
    indices=torch.stack(selected,1)
    return points.gather(1,indices[:,:,None].expand(-1,-1,3)),torch.stack(sampled_mask,1)


def local_moments(points,anchors,padding,scale):
    """Input-only local covariance descriptors; no target/radius information."""
    relative=(points[:,None,:,:]-anchors[:,:,None,:])/scale
    score=-relative.square().sum(-1)
    score=score.masked_fill(padding[:,None,:],float('-inf'))
    weights=score.softmax(-1)
    mean=torch.einsum('bmn,bmnd->bmd',weights,relative)
    centered=relative-mean[:,:,None,:]
    x,y,z=centered.unbind(-1)
    products=torch.stack((x*x,y*y,z*z,x*y,x*z,y*z),-1)
    return torch.einsum('bmn,bmnd->bmd',weights,products)


class FourierPoints(nn.Module):
    def __init__(self,width,bands=8):
        super().__init__(); self.register_buffer('frequencies',math.pi*2**torch.arange(bands,dtype=torch.float32))
        self.projection=nn.Linear(3+6*bands,width)
    def forward(self,points):
        phase=points[:,:,:,None]*self.frequencies
        features=torch.cat((points,phase.sin().flatten(2),phase.cos().flatten(2)),-1)
        return self.projection(features)


class AttentionBlock(nn.Module):
    def __init__(self,width,heads,residual=True):
        super().__init__(); self.residual=residual; self.query_norm=nn.LayerNorm(width); self.context_norm=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,heads,batch_first=True,dropout=0.)
        if residual:
            self.ff_norm=nn.LayerNorm(width)
            self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,query,context=None,padding=None,query_padding=None,bias=None):
        context=query if context is None else context
        attention_mask=None
        if bias is not None:
            attention_mask=bias[:,None].expand(-1,self.attention.num_heads,-1,-1).reshape(-1,bias.shape[-2],bias.shape[-1])
            if padding is not None:
                attention_mask=attention_mask.masked_fill(padding[:,None,None,:].expand(-1,self.attention.num_heads,bias.shape[-2],-1).reshape_as(attention_mask),float('-inf'))
                padding=None
        output=self.attention(self.query_norm(query),self.context_norm(context),self.context_norm(context),key_padding_mask=padding,attn_mask=attention_mask,need_weights=False)[0]
        if not self.residual:
            return output.masked_fill(query_padding[:,:,None],0) if query_padding is not None else output
        result=query+output; result=result+self.ff(self.ff_norm(result))
        return result.masked_fill(query_padding[:,:,None],0) if query_padding is not None else result


class GraphOccupancyAE(nn.Module):
    def __init__(self,width=96,latent_dim=16,tokens=64,heads=4,encoder_layers=2,decoder_layers=2,bands=8,variational=True,condition_dim=0,spatial_anchors=False,attention_scale=.05,decoder_query_residual=True,cross_heads=None,output_layer_norm=True,relative_feature_scale=None,geometric_moments=False):
        super().__init__(); self.tokens=tokens; self.variational=variational; self.condition_dim=condition_dim
        self.spatial_anchors=spatial_anchors;self.attention_scale=attention_scale
        self.relative_feature_scale=attention_scale if relative_feature_scale is None else relative_feature_scale
        if not math.isfinite(self.relative_feature_scale) or self.relative_feature_scale<=0:raise ValueError("Positive finite relative feature scale required")
        self.geometric_moments=geometric_moments
        self.points=FourierPoints(width,bands)
        self.condition=nn.Linear(condition_dim,width) if condition_dim else None
        self.encoder_cross=AttentionBlock(width,cross_heads or heads)
        self.encoder_self=nn.ModuleList([AttentionBlock(width,heads) for _ in range(encoder_layers)])
        self.mean=nn.Linear(width,latent_dim); self.logvar=nn.Linear(width,latent_dim)
        if not variational:self.logvar.requires_grad_(False)
        self.project=nn.Linear(latent_dim,width)
        self.decoder_self=nn.ModuleList([AttentionBlock(width,heads) for _ in range(decoder_layers)])
        self.decoder_cross=AttentionBlock(width,cross_heads or heads,residual=decoder_query_residual)
        self.output=(nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1)) if output_layer_norm else nn.Linear(width,1))
        if spatial_anchors:
            del self.decoder_cross
            self.spatial_q=nn.Linear(width,width);self.spatial_k=nn.Linear(width,width)
            self.spatial_value=nn.Linear(width,width);self.spatial_relation=nn.Linear(3,width)
            self.spatial_ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))
        if geometric_moments:
            self.moments=nn.Sequential(nn.Linear(6,width),nn.GELU(),nn.Linear(width,width))



    def encode(self,cloud,padding,conditions=None,sample=None):
        anchors,token_padding=farthest_points(cloud,padding,self.tokens)
        context=self.points(cloud); query=self.points(anchors)
        if self.geometric_moments:
            query=query+self.moments(local_moments(cloud,anchors,padding,self.attention_scale))

        if self.condition is not None:
            if conditions is None or conditions.shape!=(len(cloud),self.condition_dim) or not torch.isfinite(conditions).all():
                raise ValueError('Known finite subject conditions required')
            condition=self.condition(conditions.detach())[:,None]
            context=context+condition; query=query+condition
        bias=-torch.cdist(anchors,cloud).square()/self.attention_scale**2 if self.spatial_anchors else None
        hidden=self.encoder_cross(query,context,padding,token_padding,bias=bias)
        for layer in self.encoder_self: hidden=layer(hidden,padding=token_padding,query_padding=token_padding)
        mean=self.mean(hidden).masked_fill(token_padding[:,:,None],0)
        logvar=self.logvar(hidden).clamp(-12,8).masked_fill(token_padding[:,:,None],0)
        use_sample=self.training if sample is None else sample
        z=mean+torch.randn_like(mean)*torch.exp(.5*logvar) if self.variational and use_sample else mean
        z=z.masked_fill(token_padding[:,:,None],0)
        valid=(~token_padding)[:,:,None]
        kl=(.5*(mean.square()+logvar.exp()-1-logvar)*valid).sum()/((valid.sum()*mean.shape[-1]).clamp_min(1)) if self.variational else mean.new_zeros(())
        return {'tokens':z,'mean':mean,'logvar':logvar,'padding':token_padding,'anchors':anchors,'kl':kl}

    def prepare_decoder(self,encoded):
        hidden=self.project(encoded['tokens'])
        for layer in self.decoder_self: hidden=layer(hidden,padding=encoded['padding'],query_padding=encoded['padding'])
        return hidden

    def decode(self,queries,encoded,prepared=None):
        hidden=self.prepare_decoder(encoded) if prepared is None else prepared
        if self.spatial_anchors:
            # Anchors are part of this explicit spatial-token variant, not raw graph
            # access or a distance-to-graph occupancy formula. All tokens contribute
            # continuously; no nearest-k or capsule threshold exists in the decoder.
            embedding=self.points(queries)
            relative=queries[:,:,None,:]-encoded['anchors'][:,None,:,:]
            score=torch.einsum('bqd,bkd->bqk',self.spatial_q(embedding),self.spatial_k(hidden))/hidden.shape[-1]**.5
            score=score-relative.square().sum(-1)/self.attention_scale**2
            score=score.masked_fill(encoded['padding'][:,None,:],float('-inf'))
            weights=score.softmax(-1)
            values=torch.nn.functional.gelu(self.spatial_value(hidden)[:,None,:,:]+self.spatial_relation(relative/self.relative_feature_scale))
            output=torch.einsum('bqk,bqkd->bqd',weights,values)
            output=output+self.spatial_ff(output)
        else:
            output=self.decoder_cross(self.points(queries),hidden,encoded['padding'])
        return self.output(output).squeeze(-1)

    def forward(self,cloud,padding,queries,conditions=None):
        encoded=self.encode(cloud,padding,conditions)
        return {'logits':self.decode(queries,encoded),'kl':encoded['kl'],'encoded':encoded}
