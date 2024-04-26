# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
import torch

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)

def build_sam(
    model_type="vit_b",
    checkpoint=None,
    device="cpu",
    use_poly=True,
    add_edge=True,
    load_pl=False,
    freeze_img=False,
    freeze_mask=False,
    image_size=1024,
    upconv=False,
    remove_polyweight=False,
    selected_imgfeats=None,#用于Prompter中获取中间特征[2,4,6,7,8,9,10,11]
    **kwargs
):
    if model_type=="vit_b":
        encoder_embed_dim=768
        encoder_depth=12
        encoder_num_heads=12
        encoder_global_attn_indexes=[2, 5, 8, 11]
    elif model_type=="vit_l":
        encoder_embed_dim=1024
        encoder_depth=24
        encoder_num_heads=16
        encoder_global_attn_indexes=[5, 11, 17, 23]
    elif model_type=="vit_h":
        encoder_embed_dim=1280
        encoder_depth=32
        encoder_num_heads=16
        encoder_global_attn_indexes=[7, 15, 23, 31]
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    decoder_params = dict(num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )  
    if use_poly:
        from .modeling.mask_poly_decoder import MaskPolyDecoder
        decoder_params["add_edge"]=add_edge
        decoder_params['upconv']=upconv
        mask_decoder=MaskPolyDecoder(**decoder_params)
    else:
        mask_decoder=MaskDecoder(**decoder_params)
    img_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        selected_imgfeats=selected_imgfeats,
    )
    sam = Sam(
        image_encoder=img_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375]
    )
    if checkpoint is not None:
        state_dict=torch.load(checkpoint,map_location=device)
        if load_pl:
            state_dict=load_pl_model(state_dict)
            if remove_polyweight:#for auto mode continue training
                remove=['output_poly_upscaling','output_off_upscaling']
                state_dict=remove_keys(state_dict,remove)
        else:
            if image_size!=1024:
                state_dict=interp_weight(state_dict,image_size,encoder_global_attn_indexes)
        sam.load_state_dict(state_dict, strict=False)
    for param in sam.prompt_encoder.parameters():#
        param.requires_grad = False
    if freeze_img:
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
    if freeze_mask:
        for param in sam.mask_decoder.parameters():
            param.requires_grad = False
    return sam
import torch.nn.functional as F
def interp_weight(state_dict,image_size,encoder_global_attn_indexes):
    key='image_encoder.pos_embed'
    weight = state_dict[key]
    #[1, 64, 64, 768]->[1, image_size/16, image_size/16, 768]
    pos_size=int(image_size/16)
    state_dict[key]=F.interpolate(weight.permute(0,3,1,2),
        size=(pos_size, pos_size), mode='bilinear', align_corners=True).permute(0,2,3,1)
    for key in state_dict.keys():
        if ('rel_pos_h' in key or 'rel_pos_w' in key) \
            and any(str(n) in key for n in encoder_global_attn_indexes):#2 5 8 11 sam_b
            weight = state_dict[key]#[127,64]
            # 更改形状以适应 interpolate 函数:
            weight_reshaped = weight.unsqueeze(0).permute(0, 2, 1)#更改为 [1, 64, 127]
            #[1, 64, 127]->[1, 64, 27]->[27, 64]
            state_dict[key]=F.interpolate(weight_reshaped,
                size=int(image_size/8-1), mode='linear', align_corners=True).permute(0, 2, 1).squeeze(0)
    return state_dict
def load_pl_model(state_dict,remove_prefix='sam_model.'):
    state_dict = state_dict['state_dict']
    # 删除前缀：
    new_state_dict = {k.replace(remove_prefix, ''): v for k, v in state_dict.items()}
    return new_state_dict

def remove_keys(state_dict,keys):
    remove=[]
    for key in state_dict.keys():
        if any(k in key for k in keys):#权重名中包含keys中的任意一个字符串
            remove.append(key)
    for key in remove:
        del state_dict[key]
    return state_dict