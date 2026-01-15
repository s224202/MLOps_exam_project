import hydra
from omegaconf import DictConfig  # ja, den her er “OmegaConf”, men kun som type


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print kun det, I vil se (ingen config-dump)
    print("env.data_root:", cfg.env.paths.data_root)
    print("runtime.device:", cfg.env.runtime.device)
    print("model:", cfg.model.name)
    print("epochs:", cfg.training.epochs)


if __name__ == "__main__":
    main()
