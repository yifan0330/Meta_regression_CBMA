import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/data", config_name="datasets")
def main(config: DictConfig):
    from data_generation.Data_simulation import Simulation
    Simulation.Poisson(config)

    return 

if __name__ == "__main__":
    main()