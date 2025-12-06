import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"--- 正在运行实验: {cfg.experiment_name} ---")

    # 1. 实例化 DataModule
    print(f"初始化数据: {cfg.data._target_}")
    dm = hydra.utils.instantiate(cfg.data)

    # 2. 实例化 Model
    print(f"初始化模型: {cfg.model._target_} (LR: {cfg.model.learning_rate})")
    model = hydra.utils.instantiate(cfg.model)

    # 3. 实例化 Trainer
    # 这里我们手动加一个 RichProgressBar 让控制台输出好看点
    print("初始化 Trainer...")
    trainer = hydra.utils.instantiate(
        cfg.train, 
        callbacks=[RichProgressBar()]
    )

    # 4. 开始训练
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
