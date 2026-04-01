from pydantic import BaseModel, Field
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Literal

class MainConfig(BaseModel):
    lr: float = Field(gt=0, lt=1)
    epochs: int = Field(gt=0)
    n_rand: int = Field(gt=0)
    n_samples: int = Field(gt=0)
    near: float
    far: float
    dataset: Literal["tless", "tiny"]
    directory: str



class GSConfig(BaseModel):
    lr: float = Field(gt=0, lt=1)
    max_steps: int = Field(gt=0)
    camera_width: int = Field(gt=0)
    camera_height: int = Field(gt=0)
    num_points: int = Field(gt=0)
    ssim_coefficient: float
    dataset: Literal["tless", "tiny"]
    directory: str
    name: str
    save_directory: str