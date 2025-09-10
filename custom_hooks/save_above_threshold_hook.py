from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS  # 导入 HOOKS 注册器

@HOOKS.register_module()
class SaveAboveThresholdHook(Hook):
    def __init__(self, metric_key, threshold, save_dir, rule='greater'):
        """
        Args:
            metric_key (str): 要监控的指标名称，如 "mAP_0.25" 或 "mAP_0.50"。
            threshold (float): 阈值，比如 0.77 或 0.65。
            save_dir (str): 保存检查点的目录。
            rule (str): 指标比较规则，'greater' 表示指标越大越好。
        """
        self.metric_key = metric_key
        self.threshold = threshold
        self.save_dir = save_dir
        self.rule = rule
        import os
        os.makedirs(self.save_dir, exist_ok=True)

    def after_val_epoch(self, runner, **kwargs):
        # 从 kwargs 中获取验证指标
        metrics = kwargs.get('metrics', None)
        if metrics is None:
            runner.logger.warning("No metrics provided in after_val_epoch hook.")
            return

        # 输出所有指标键（调试用）
        runner.logger.info(f"Validation metrics: {list(metrics.keys())}")

        if self.metric_key not in metrics:
            runner.logger.warning(f"Metric '{self.metric_key}' not found in validation metrics.")
            return

        curr_val = metrics[self.metric_key]
        runner.logger.info(f"Current {self.metric_key}: {curr_val:.4f}")

        condition_met = False
        if self.rule == 'greater' and curr_val > self.threshold:
            condition_met = True
        elif self.rule == 'less' and curr_val < self.threshold:
            condition_met = True

        if condition_met:
            epoch = runner.epoch + 1  # 假设 epoch 从 0 开始
            ckpt_name = f"{self.metric_key}_epoch{epoch}_val{curr_val:.4f}.pth"
            ckpt_path = os.path.join(self.save_dir, ckpt_name)
            runner.logger.info(f"Threshold met for {self.metric_key} ({curr_val:.4f} > {self.threshold}). Saving checkpoint: {ckpt_path}")
            runner.save_checkpoint(ckpt_path)
