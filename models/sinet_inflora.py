import torch
import torch.nn as nn
import copy
from functools import partial

from models.vit_inflora import VisionTransformer, PatchEmbed, Block, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.zoo import CodaPrompt


class ViT_lora_co(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
<<<<<<< HEAD
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64,**kwargs ):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank,**kwargs)

=======
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64, **kwargs):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
                         embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank, **kwargs)
>>>>>>> 227a2fa (sam)

    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk == i,
                    get_feat=get_feat, get_cur_feat=get_cur_feat)

        x = self.norm(x)

        return x, prompt_loss


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError(
            'features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = resolve_pretrained_cfg(variant)

    print('---debug----')
    print(pretrained_cfg)

    # default_num_classes = pretrained_cfg['num_classes']
    default_num_classes = pretrained_cfg.num_classes
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

<<<<<<< HEAD
    
    # ✅ 提前处理这个字段
    custom_load = ('npz' in pretrained_cfg.url) if pretrained_cfg.url else False
=======
    # ✅ 提前处理这个字段
    custom_load = (
        'npz' in pretrained_cfg.url) if pretrained_cfg.url else False
>>>>>>> 227a2fa (sam)
    # pretrained_cfg['custom_load'] = custom_load

    # 替换成属性赋值
    if hasattr(pretrained_cfg, "custom_load"):
        pretrained_cfg.custom_load = custom_load
    # ✅ 如果是 npz，不要触发 timm 的 torch.load
    # timm_pretrained = pretrained and not custom_load

    print('---debug:---- ')
<<<<<<< HEAD
    print(f' build_model_with_cfg(pretrained={pretrained},pretrained_cfg={pretrained_cfg},pretrained_custom_load={custom_load}) ')
    
    model = build_model_with_cfg(
        ViT_lora_co, variant, 
        pretrained= pretrained,  
=======
    print(
        f' build_model_with_cfg(pretrained={pretrained},pretrained_cfg={pretrained_cfg},pretrained_custom_load={custom_load}) ')

    model = build_model_with_cfg(
        ViT_lora_co, variant,
        pretrained=pretrained,
>>>>>>> 227a2fa (sam)
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load=custom_load,
<<<<<<< HEAD
            **kwargs)
=======
        **kwargs)
>>>>>>> 227a2fa (sam)

    return model

    # model = build_model_with_cfg(
    #     ViT_lora_co, variant, pretrained,
    #     pretrained_cfg=pretrained_cfg,
    #     representation_size=repr_size,
    #     pretrained_filter_fn=checkpoint_filter_fn,
    #     pretrained_custom_load='npz' in pretrained_cfg['url'],#     **kwargs)

    # model = build_model_with_cfg(
    # ViT_lora_co, variant, pretrained,
    # pretrained_cfg=pretrained_cfg,
    # representation_size=repr_size,
    # pretrained_filter_fn=checkpoint_filter_fn,
    # pretrained_custom_load='npz' in pretrained_cfg.url,# **kwargs
    # )

    # # ❗️提前获取 custom_load 标志
    # custom_load = 'npz' in pretrained_cfg.url if pretrained_cfg.url else False

    # # ❗️剔除不支持的 kwargs
    # kwargs.pop('pretrained_custom_load', None)

    # # 构建模型
    # model = build_model_with_cfg(
    #     ViT_lora_co, variant, pretrained,
    #     pretrained_cfg=pretrained_cfg,
    #     representation_size=repr_size,
    #     pretrained_filter_fn=checkpoint_filter_fn,
    #     # 只作为参数传给 builder 函数，不传进模型构造器
    #     pretrained_custom_load=custom_load,
    #     **kwargs
    # )
    # # 构建构造函数的 partial，不传非法参数
    # custom_load = 'npz' in pretrained_cfg.url if pretrained_cfg.url else False

    # model_cls = partial(ViT_lora_co, **kwargs)

    # model = build_model_with_cfg(
    #     model_cls, variant, pretrained,
    #     pretrained_cfg=pretrained_cfg,
    #     representation_size=repr_size,
    #     pretrained_filter_fn=checkpoint_filter_fn,
    #     pretrained_custom_load=custom_load
    # )


    # model = build_model_with_cfg(
    #     ViT_lora_co, variant, pretrained,
    #     pretrained_cfg=pretrained_cfg,
    #     representation_size=repr_size,
    #     pretrained_filter_fn=checkpoint_filter_fn,
    #     pretrained_custom_load='npz' in pretrained_cfg['url'],#     **kwargs)


    # model = build_model_with_cfg(
    # ViT_lora_co, variant, pretrained,
    # pretrained_cfg=pretrained_cfg,
    # representation_size=repr_size,
    # pretrained_filter_fn=checkpoint_filter_fn,
    # pretrained_custom_load='npz' in pretrained_cfg.url,# **kwargs
    # )

    # # ❗️提前获取 custom_load 标志
    # custom_load = 'npz' in pretrained_cfg.url if pretrained_cfg.url else False

    # # ❗️剔除不支持的 kwargs
    # kwargs.pop('pretrained_custom_load', None)

    # # 构建模型
    # model = build_model_with_cfg(
    #     ViT_lora_co, variant, pretrained,
    #     pretrained_cfg=pretrained_cfg,
    #     representation_size=repr_size,
    #     pretrained_filter_fn=checkpoint_filter_fn,
    #     # 只作为参数传给 builder 函数，不传进模型构造器
    #     pretrained_custom_load=custom_load,
    #     **kwargs
    # )
    # # 构建构造函数的 partial，不传非法参数
    # custom_load = 'npz' in pretrained_cfg.url if pretrained_cfg.url else False

    # model_cls = partial(ViT_lora_co, **kwargs)

    # model = build_model_with_cfg(
    #     model_cls, variant, pretrained,
    #     pretrained_cfg=pretrained_cfg,
    #     representation_size=repr_size,
    #     pretrained_filter_fn=checkpoint_filter_fn,
    #     pretrained_custom_load=custom_load
    # )

    

    

    

    



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

<<<<<<< HEAD
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=args["total_sessions"], rank=args["rank"])


=======
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12,
                            num_heads=12, n_tasks=args["total_sessions"], rank=args["rank"])

>>>>>>> 227a2fa (sam)
        # 判断是否跳过 timm 自带预训练逻辑，改为加载训练后模型权重
        if args.get("reload_model", False):
            # 不使用 ImageNet21k 预训练，构建裸模型
            self.image_encoder = _create_vision_transformer(
                'vit_base_patch16_224_in21k', pretrained=False, **model_kwargs)
        else:
            # 默认行为：加载 timm 的 ImageNet21k 权重
            self.image_encoder = _create_vision_transformer(
                'vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)

<<<<<<< HEAD

=======
>>>>>>> 227a2fa (sam)
        # 分类器池：每个任务一个头
        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for _ in range(args["total_sessions"])
        ])
        self.classifier_pool_backup = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for _ in range(args["total_sessions"])
        ])

        # 当前任务编号
        self.numtask = 0

        # ✅ 若 reload_model，则手动加载训练好的模型参数
        if args.get("reload_model", False):
            assert "pretrained_ckpt" in args, "reload_model=True 但未提供 pretrained_ckpt 参数！"
<<<<<<< HEAD
            state_dict = torch.load(args["pretrained_ckpt"], map_location="cpu")

            # 可选择只加载 image_encoder，也可加载整体 _network
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
=======
            state_dict = torch.load(
                args["pretrained_ckpt"], map_location="cpu")

            # 可选择只加载 image_encoder，也可加载整体 _network
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False)
>>>>>>> 227a2fa (sam)
            print(f"[SiNet] Loaded from checkpoint: {args['pretrained_ckpt']}")
            print(f"[SiNet] Missing keys: {missing_keys}")
            print(f"[SiNet] Unexpected keys: {unexpected_keys}")

<<<<<<< HEAD


=======
>>>>>>> 227a2fa (sam)
    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None):
        if task == None:
            image_features, _ = self.image_encoder(image, self.numtask-1)
        else:
            image_features, _ = self.image_encoder(image, task)
        image_features = image_features[:, 0, :]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features, prompt_loss = self.image_encoder(
            image, task_id=self.numtask-1, get_feat=get_feat, get_cur_feat=get_cur_feat)
        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.size(0), -1)
        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }

<<<<<<< HEAD
    def interface(self, image, task_id = None):
        image_features, _ = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)
=======
    def interface(self, image, task_id=None):
        image_features, _ = self.image_encoder(
            image, task_id=self.numtask-1 if task_id is None else task_id)
>>>>>>> 227a2fa (sam)

        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.size(0), -1)

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits, 1)
        return logits

    def interface1(self, image, task_ids):
        logits = []
        for index in range(len(task_ids)):
            image_features, _ = self.image_encoder(
                image[index:index+1], task_id=task_ids[index].item())
            image_features = image_features[:, 0, :]
            image_features = image_features.view(image_features.size(0), -1)

            logits.append(
                self.classifier_pool_backup[task_ids[index].item()](image_features))

        logits = torch.cat(logits, 0)
        return logits

    def interface2(self, image_features):

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits, 1)
        return logits

    def update_fc(self, nb_classes):
        self.numtask += 1

    def classifier_backup(self, task_id):
        self.classifier_pool_backup[task_id].load_state_dict(
            self.classifier_pool[task_id].state_dict())

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
