import comet_ml  # type: ignore[import]
import hydra
from dataprocessing.runner import VitalRunner
from omegaconf import DictConfig


class DidacticRunner(VitalRunner):
    """Entry-point for a `VitalRunner` that adds the `dafted` config dir to the Hydra search path."""

    @staticmethod
    @hydra.main(version_base=None, config_path="config", config_name="default")
    def run_system(cfg: DictConfig) -> None:  # noqa: D102
        VitalRunner.run_system(cfg)


def main():
    """Run the script."""
    DidacticRunner.main()


if __name__ == "__main__":
    main()
