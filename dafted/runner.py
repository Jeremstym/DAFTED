import logging

import comet_ml  # type: ignore[import]
import hydra
from dafted.core.runner import VitalRunner
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DidacticRunner(VitalRunner):
    """Entry-point for a `VitalRunner` that adds the `dafted` config dir to the Hydra search path."""

    @staticmethod
    @hydra.main(version_base=None, config_path="config", config_name="default")
    def run_system(cfg: DictConfig) -> None:  # noqa: D102
        from dafted.data.schema import get_schema_path

        logger.info(f"Using the dataset schema from '{get_schema_path()}' (set `DAFTED_SCHEMA` to change it)")
        VitalRunner.run_system(cfg)


def main():
    """Run the script."""
    DidacticRunner.main()


if __name__ == "__main__":
    main()
