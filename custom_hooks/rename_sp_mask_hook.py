# custom_hooks/rename_sp_mask_hook.py

from mmengine.hooks import Hook
from mmengine.runner import Runner

class RenameSPMaskHook(Hook):
    """在每个 train iter 之前，将 ds.gt_pts_seg.sp_pts_mask 复制到 sp_mask."""
    def before_train_iter(self, runner: Runner) -> None:
        # mmengine 会把这个 batch 的 data_samples 挂到 runner.data_samples
        data_samples = getattr(runner, 'data_samples', None)
        if data_samples is None:
            return
        for ds in data_samples:
            pts_seg = getattr(ds, 'gt_pts_seg', None)
            if pts_seg is None:
                continue
            # 如果有 sp_pts_mask，就把它赋给 sp_mask
            if hasattr(pts_seg, 'sp_pts_mask'):
                pts_seg.sp_mask = pts_seg.sp_pts_mask
