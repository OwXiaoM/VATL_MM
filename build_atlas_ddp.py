import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.inr_decoder import INR_Decoder
from data_loading.dataset import Data
from utils import *

class AtlasBuilderDDP:
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        self.rank = args['rank']
        # 【关键修改】获取 local_rank 用于绑定 GPU，防止所有进程抢占 GPU 0
        self.local_rank = args.get('local_rank', self.rank)
        self.world_size = args['world_size']
        
        self.loss_criterion = Criterion(args).to(self.device)
        self._init_atlas_training()
        self.train_on_data()

    def broadcast_latents_from_rank0(self):
        """
        【新增】手动广播 Latents。
        因为 Latents 是外部 Parameter，不属于 DDP 模型的一部分，
        必须手动确保所有卡在训练开始时的 Latents 是完全一致的。
        """
        if self.rank == 0:
            print("[DDP] Broadcasting initial latents...")
        
        # 广播训练集 Latents
        if 'train' in self.latents:
            dist.broadcast(self.latents['train'].data, src=0)
        
        # 广播变换网络参数 (如果有)
        if 'train' in self.transformations and self.args['inr_decoder']['tf_dim'] > 0:
            dist.broadcast(self.transformations['train'].data, src=0)

    def train_on_data(self):
        # 1. 确保所有卡的起点一致
        self.broadcast_latents_from_rank0()
        
        # 2. 初始验证 (仅 Rank 0 执行，或者根据需要调整)
        # 建议：如果是 Resume，可以跑一次；如果是从头训练，可以跳过节省时间
        if len(self.args['load_model']['path']) > 0 and self.rank == 0:
             self.validate(epoch_train=0)
        
        # 【关键修改】设置屏障，等待 Rank 0 验证完成，防止其他卡抢跑导致不同步
        dist.barrier()
            
        loss_hist_epochs = []
        start_time = time.time()
        
        for epoch in range(self.args['epochs']['train']):
            # DDP Sampler 必须调用 set_epoch 才能 shuffle
            self.dataloaders['train'].sampler.set_epoch(epoch)
            
            # 执行训练
            loss = self.train_epoch(epoch, split='train')
            loss_hist_epochs.append(loss)
            
            # 记录日志 (仅 Rank 0)
            if self.rank == 0:
                print(f"Training: Epoch: {epoch}, Loss: {loss:.4f}, Total Time: {time.time() - start_time:.2f}s")
                
                # 保存模型 & 验证
                if epoch > 0 and (epoch % self.args['save_every'] == 0 or epoch == self.args['epochs']['train'] - 1):
                    self.save_state(epoch)
                
                if epoch > 0 and (epoch % self.args['validate_every'] == 0 or epoch == self.args['epochs']['train'] - 1):
                    self.validate(epoch)

            # 【关键修改】每个 Epoch 结束后同步一次，确保步调一致
            dist.barrier()
            
            # 更新学习率
            self._update_scheduler(split='train')

    def train_epoch(self, epoch, split='train'):
        self.inr_decoder[split].train()
        loss_hist_batches = []
        
        for i, batch in enumerate(self.dataloaders[split]):
            loss = self.train_batch(batch, epoch, split)
            loss_hist_batches.append(loss)
            
            # 仅 Rank 0 打印进度
            if self.rank == 0 and (i % 10 == 0): 
                 print(f"Split: {split}, Epoch: {epoch}, Batch: {i}/{len(self.dataloaders[split])}, Loss: {loss:.4f}")
                 
        return np.mean(loss_hist_batches)

    def train_batch(self, batch, epoch, split='train'):
        loss_hist_samples = []
        n_smpls = self.args['n_samples']
        seg_weight = self.args['optimizer']['seg_weight'] if split == 'train' else 0.0
        
        coords_batch, values_batch, conditions_batch, idx_df_batch = to_device(batch, self.device)
        
        # 注意：这里的 idx_df_batch.shape[0] 是由 dataset.py 中的采样点数决定的。
        sample_iterator = range(0, idx_df_batch.shape[0], n_smpls)
        
        for i, smpls in enumerate(sample_iterator):
            self.optimizers[split].zero_grad()
            
            coords = coords_batch[smpls:smpls + n_smpls]
            values = values_batch[smpls:smpls + n_smpls]
            
            # 处理索引
            idx_df = idx_df_batch[smpls:smpls + n_smpls].squeeze()
            if idx_df.ndim == 0: idx_df = idx_df.unsqueeze(0) 
            idx_df = idx_df.long()

            conditions = conditions_batch[smpls:smpls + n_smpls] if split == 'train' else self.conditions_val[idx_df]

            with torch.autocast(device_type='cuda', enabled=self.args['amp']):
                # DDP 模型 Forward
                ret = self.inr_decoder[split](
                    coords, 
                    self.latents[split], 
                    conditions,
                    self.transformations[split][idx_df], 
                    idcs_df=idx_df
                )
                
                if isinstance(ret, tuple):
                    values_p, aux_loss = ret
                else:
                    values_p = ret
                    aux_loss = None

                # 【修复】这里返回的是一个字典 {'total': ..., 'seg': ...}
                loss_dict = self.loss_criterion(values_p, values, self.transformations[split][idx_df], 
                                           moe_loss=aux_loss, seg_weight=seg_weight)
                
                # 【修复】必须提取 'total' 标量 Tensor 才能 backward
                loss = loss_dict['total']

            # Backward & Optimizer Step
            if self.args['amp']:    
                self.grad_scalers[split].scale(loss).backward()
                self.grad_scalers[split].step(self.optimizers[split])
                self.grad_scalers[split].update()
            else:
                loss.backward()
                self.optimizers[split].step()
                
            loss_hist_samples.append(loss.item())
            
        return np.mean(loss_hist_samples)

    def validate(self, epoch_train):
        """
        验证函数。注意：仅 Rank 0 调用此函数。
        """
        # 保存状态
        self.save_state(epoch_train)
        
        # 生成图谱 (如果配置开启)
        if self.args['generate_cond_atlas']: 
            print(f"[Rank {self.rank}] Generating Atlas...", flush=True)
            self.generate_atlas(epoch_train, n_max=100)
            print(f"[Rank {self.rank}] Atlas Generation Done.", flush=True)

        print(f"[Rank {self.rank}] Starting inference for Epoch {epoch_train}...", flush=True)
        
        num_train = len(self.datasets['train'])
        # 调试建议：可以先只跑几个样本，确保流程通畅
        train_indices = [0, 2, 3] if num_train > 3 else list(range(num_train))
        
        # 运行推断
        metrics_train = self.generate_subjects_from_df(idcs_df=train_indices, epoch=epoch_train, split='train')
        
        # 记录指标
        log_metrics(self.args, metrics_train, epoch_train, df=self.datasets['train'].df, split='train')
        print(f"[Rank {self.rank}] Validation Finished.", flush=True)

    def save_state(self, epoch, split='train'):
        # 仅 Rank 0 保存
        if self.rank != 0: return 
        
        if self.args['save_model']:
            log_dir = self.args['output_dir']
            
            # 【关键修改】解包 DDP 模型，防止 state_dict key 出现 'module.' 前缀
            model_to_save = self.inr_decoder[split].module if isinstance(self.inr_decoder[split], DDP) else self.inr_decoder[split]
            
            torch.save({
                'epoch': epoch,
                'latents': self.latents[split].cpu(),
                'transformations': self.transformations[split].cpu(),
                'inr_decoder': model_to_save.state_dict(),
                'tsv_file': self.datasets[split].tsv_file,
                'dataset_df': self.datasets[split].df,
                'args': self.args
            }, os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Saved model state to {os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth")}')

    def _init_inr(self, state_dict=None, split='train'):
        self.args['inr_decoder']['cond_dims'] = sum([self.args['dataset']['conditions'][c] 
                                                     for c in self.args['dataset']['conditions']])
        
        model = INR_Decoder(self.args, self.device).to(self.device)
        
        if state_dict is not None:
            # 兼容加载 DDP 或非 DDP 权重
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)

        if split == 'train':
            # 【关键修改】标准 DDP 初始化
            # 1. device_ids 绑定到 local_rank
            # 2. find_unused_parameters=True (MoE 必须)
            self.inr_decoder[split] = DDP(
                model, 
                device_ids=[self.local_rank], 
                output_device=self.local_rank, 
                find_unused_parameters=True
            )
        else:
            self.inr_decoder[split] = model

    def _init_dataloading(self, tsv_file=None, df_loaded=None, split='train'):
        tsv_file = pd.read_csv(self.args['dataset']['tsv_file'], sep='\t') if tsv_file is None else tsv_file
        self.datasets[split] = Data(self.args, tsv_file, split=split, df_loaded=df_loaded)

        sampler = None
        shuffle = (split == 'train')
        if split == 'train':
            # DDP Sampler
            sampler = DistributedSampler(self.datasets[split], 
                                         num_replicas=self.world_size, 
                                         rank=self.rank, 
                                         shuffle=True) 
            shuffle = False # Sampler 会处理 shuffle

        # 【关键修改】DataLoader 优化
        num_workers = int(self.args['dataset'].get('num_workers', 4))
        # 开启 persistent_workers，避免每个 Epoch 重建进程
        persistent_workers = (num_workers > 0)

        dataloader_kwargs = dict(
            dataset=self.datasets[split],
            batch_size=self.args['batch_size'],
            num_workers=num_workers,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.datasets[split].collate_fn,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
        
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = 2

        self.dataloaders[split] = DataLoader(**dataloader_kwargs)
        if self.rank == 0:
            print(f"Initialized DDP dataloader for {split} with {len(self.datasets[split])} subjects.")

    def _init_atlas_training(self):
        self._seed()
        self.datasets = {}
        self.dataloaders = {}
        self.inr_decoder = {}
        self.latents = {}
        self.conditions_val = {}
        self.transformations = {} # 虽然这里初始化了字典，但里面没数据
        self.optimizers = {}
        self.schedulers = {}
        self.grad_scalers = {}

        # 1. 加载数据
        self._init_dataloading(split='train')
        
        # 2. 加载/初始化模型状态
        state_dict = None
        if len(self.args['load_model']['path']) > 0:
            print(f"Loading model from {self.args['load_model']['path']}")
            checkpoint = torch.load(self.args['load_model']['path'], map_location=self.device)
            state_dict = checkpoint['inr_decoder']
            # 注意：如果需要 Resume transformations，这里还需要加载 checkpoint['transformations']
        
        # 3. 初始化 INR 模型
        self._init_inr(state_dict, split='train')
        
        # 4. 初始化 Latents / Transformations / 优化器
        self._init_latents(split='train')
        self._init_transformations(split='train') # 【必须补上这行！】
        self._init_optimizer(split='train')
        
    def _init_latents(self, lats=None, split='train'):
        n_subjects = len(self.datasets[split])
        latent_dim = self.args['inr_decoder']['latent_dim']
        
        # 【关键修复】判断 latent_dim 是 list 还是 int
        # 配置文件里通常写的是 [256]，需要解包
        if isinstance(latent_dim, list) or isinstance(latent_dim, tuple):
            shape = (n_subjects, *latent_dim) # 变成 (100, 256)
        else:
            shape = (n_subjects, latent_dim)  # 变成 (100, 256)
        
        # 初始化
        if lats is None:
            lats = torch.normal(0, 0.01, size=shape, device=self.device)
        else:
            lats = lats.to(self.device)
            
        self.latents[split] = nn.Parameter(lats)
        
        # 初始化 Val conditions
        if split == 'val': 
            # cond_dims 通常是求和后的 int，所以这里直接用没问题
            shape_cond_val = (len(self.datasets['val']), self.args['inr_decoder']['cond_dims'])
            self.conditions_val = nn.Parameter(torch.normal(0, 0.01, size=shape_cond_val).to(self.device))

    def _init_transformations(self, tfs=None, split='train'):
        tf_dim = self.args['inr_decoder']['tf_dim']
        
        # 【安全修复】防止 tf_dim 也是 list (例如 [6])
        if isinstance(tf_dim, list) or isinstance(tf_dim, tuple):
            tf_dim = tf_dim[0]
            
        # 至少 6 维 (Rigid需要6参数，即使 tf_dim=0 代码逻辑也预留了空间)
        shape = (len(self.datasets[split]), max(tf_dim, 6)) 
        
        if tfs is None:
            tfs = torch.zeros(shape, device=self.device)
        else:
            tfs = tfs.to(self.device)
            
        # 如果 tf_dim > 0 则是可学习参数，否则是固定为0的张量
        self.transformations[split] = nn.Parameter(tfs) if tf_dim > 0 else tfs

    def _init_optimizer(self, split='train'):
        params = [
            {'name': f'latents_{split}', 'params': self.latents[split], 
             'lr': self.args['optimizer']['lr_latent'], 
             'weight_decay': self.args['optimizer']['latent_weight_decay']}
        ]
        
        if split in self.transformations:
            params.append({
                'name': f'transformations_{split}', 'params': self.transformations[split], 
                'lr': self.args['optimizer']['lr_tf'], 
                'weight_decay': self.args['optimizer']['tf_weight_decay']
            })
        
        if split == 'train': 
            params.append({
                'name': f'inr_decoder', 'params': self.inr_decoder[split].parameters(), 
                'lr': self.args['optimizer']['lr_inr'], 
                'weight_decay': self.args['optimizer']['inr_weight_decay']
            })
        
        self.optimizers[split] = optim.AdamW(params)
        self.grad_scalers[split] = GradScaler() if self.args['amp'] else None
        
        if self.args['optimizer']['scheduler']['type'] == 'cosine': 
            self.schedulers[split] = CosineAnnealingLR(
                self.optimizers[split], 
                T_max=self.args['epochs'][split], 
                eta_min=self.args['optimizer']['scheduler']['eta_min']
            )
        else: 
            self.schedulers[split] = None

    def _update_scheduler(self, split='train'):
        if self.schedulers[split] is not None: 
            self.schedulers[split].step()

    def _seed(self):
        seed = self.args['seed'] + self.rank # 不同 Rank 使用不同种子 (虽然 Sampler 会处理，但为了保险)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    # --- Inference Helpers (复用自原代码，稍作适配) ---
    # === 放在 AtlasBuilderDDP 类内部 ===
    def _get_raw_model(self, split='train'):
        """
        安全地获取原始模型，无论是 DDP 包装的还是普通的。
        用于访问自定义方法如 .inference()
        """
        model = self.inr_decoder[split]
        if isinstance(model, DDP):
            return model.module
        return model

    def generate_atlas(self, epoch=0, n_max=100):
        """
        [DDP Version]
        生成时空图谱 (Spatio-Temporal Atlas)。
        """
        # 【关键】获取 Raw Model
        raw_model = self._get_raw_model('train')
        raw_model.eval()
        
        print(f"Generating atlases (depending on resolution and number of atlases this may take some time) ...\n", flush=True)
        
        # 生成标准空间网格
        grid_coords, grid_shape, affine = generate_world_grid(self.args, device=self.device)
        temp_steps = self.args['atlas_gen']['temporal_values']
        atlas_list = []
        
        with torch.no_grad():
            for temp_step in temp_steps:
                # 1. 计算归一化的时间/年龄条件
                temp_step_normed = normalize_condition(self.args, 'scan_age', temp_step)
                
                # 计算平均 Latent (Mean Latent)
                mean_latent = self.get_mean_latent('scan_age', temp_step_normed, n_max=n_max)
                
                # 2. 生成其他条件组合 (如 Sex, Disease)
                condition_vectors = generate_combinations(self.args, self.args['atlas_gen']['conditions'])
                
                cond_list = []
                for c_v in condition_vectors:
                    # 【逻辑修复】拼接 scan_age
                    if self.args['dataset']['conditions'].get('scan_age', False):
                        # 注意：temp_step_normed 是 float，直接拼接到 list 中
                        c_v = [temp_step_normed] + c_v
                    
                    # 转 Tensor
                    c_v = torch.tensor(c_v, dtype=torch.float32).to(self.device)
                    
                    # 【关键】调用 raw_model.inference
                    values_p = raw_model.inference(
                        grid_coords, 
                        mean_latent, 
                        c_v, 
                        grid_shape, 
                        None # Atlas 生成通常不需要空间变换 (Rigid/Affine)
                    )
                    
                    # 处理分割结果 (Argmax -> Onehot 或者直接保留)
                    # 假设最后一维是通道/类别
                    seg = values_p[:, :, :, -1]
                    seg[seg==4] = 0 # 示例：清除特定类别，视具体任务而定
                    values_p[:, :, :, -1] = seg
                    
                    cond_list.append(values_p.detach().cpu())
                    torch.cuda.empty_cache()
                    
                atlas_list.append(torch.stack(cond_list, dim=-1))
                
        atlas_list = torch.stack(atlas_list, dim=-1) 
        
        # 保存
        save_atlas(self.args, atlas_list, affine, temp_steps, condition_vectors, epoch=epoch)

    def generate_subjects_from_df(self, idcs_df=None, epoch=0, split='train'):
        """
        [DDP Version]
        在原生空间生成受试者图像 (Native Space Inference)。
        """
        import nibabel as nib 
        metrics = []
        
        # 【关键】获取 Raw Model 以调用 .inference()
        # DDP 对象没有 inference 方法，必须通过 .module 访问
        raw_model = self._get_raw_model(split)
        raw_model.eval() # 确保进入评估模式
        
        # 内部辅助函数：生成原生网格
        def generate_native_grid(header_nii, world_bbox):
            # 1. 获取原图 shape 和 affine
            shape = header_nii.shape
            affine = header_nii.affine
            
            # 2. 生成网格索引 (i, j, k)
            i = torch.arange(0, shape[0], device=self.device)
            j = torch.arange(0, shape[1], device=self.device)
            k = torch.arange(0, shape[2], device=self.device)
            grid = torch.meshgrid(i, j, k, indexing='ij')
            grid_coords_idx = torch.stack(grid, dim=-1).reshape(-1, 3).float()
            
            # 3. 索引 -> 物理坐标 (P = A * idx)
            affine_torch = torch.tensor(affine, dtype=torch.float32, device=self.device)
            ones = torch.ones((grid_coords_idx.shape[0], 1), device=self.device)
            grid_coords_homo = torch.cat([grid_coords_idx, ones], dim=1)
            grid_coords_phys = (affine_torch @ grid_coords_homo.T).T[:, :3]
            
            # 4. 归一化 (逻辑需与 Dataset 保持一致)
            # 计算几何中心
            img_center_index = torch.tensor(shape, device=self.device) / 2.0
            center_homo = torch.cat([img_center_index, torch.tensor([1.0], device=self.device)])
            geometric_center = (affine_torch @ center_homo)[:3]
            
            # 坐标中心化 & 缩放
            grid_coords_norm = grid_coords_phys - geometric_center
            wb_torch = torch.tensor(world_bbox, dtype=torch.float32, device=self.device)
            grid_coords_norm = grid_coords_norm / (wb_torch / 2.0)
            
            return grid_coords_norm, list(shape), affine

        # 遍历需要验证的样本索引
        for idx_df in idcs_df:
            # 获取样本信息
            df_row_dict = self.datasets[split].df.iloc[idx_df].to_dict()
            ref_mod_path = df_row_dict[self.args['dataset']['modalities'][0]]
            ref_nii = nib.load(ref_mod_path)
            
            # 生成坐标网格
            grid_coords, grid_shape, affine = generate_native_grid(
                ref_nii, 
                self.args['dataset']['world_bbox']
            )
            
            with torch.no_grad():
                # 准备变换参数和 Latents
                transformations = self.transformations[split][idx_df, None]
                conditions = self.datasets[split].load_conditions(df_row_dict).to(self.device)
                
                # 【关键】调用 raw_model.inference
                volume_inf = raw_model.inference(
                    grid_coords, 
                    self.latents[split][idx_df:idx_df+1], 
                    conditions, 
                    grid_shape, 
                    transformations
                )
            
            # 计算指标或保存图像
            if self.args['compute_metrics']:
                metrics.append(compute_metrics(self.args, volume_inf, affine, df_row_dict, epoch, split))
            elif self.args['save_imgs'][split]:
                save_subject(self.args, volume_inf, affine, df_row_dict, epoch, split)
        
        return metrics

# 辅助函数：将 batch 数据移到 GPU
def to_device(batch, device):
    coords, values, conditions, idx_df = batch
    return coords.to(device), values.to(device), conditions.to(device), idx_df.to(device)