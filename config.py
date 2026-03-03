from pydantic import BaseModel, Field
import hydra
from omegaconf import DictConfig, OmegaConf

class MainConfig(BaseModel):
    lr: float = Field(gt=0, lt=1)
    epochs: int = Field(gt=0)
    n_rand: int = Field(gt=0)
    n_samples: int = Field(gt=0)
    near: float
    far: float
    directory: str