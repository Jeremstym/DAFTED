from dataprocessing import get_vital_root

from dafted.results.cardinal.utils.temporal_metrics import TemporalMetrics
from dafted.results.cardinal.utils.time_series_attributes import TimeSeriesAttributesMixin


class ImageTemporalMetrics(TimeSeriesAttributesMixin, TemporalMetrics): # type: ignore[misc]
    """Class that computes temporal coherence metrics on image time-series attributes."""

    desc = f"seg_{TemporalMetrics.desc}"
    default_attribute_statistics_cfg = get_vital_root() / "data/camus/statistics/image_attr_stats.yaml"


def main():
    """Run the script."""
    ImageTemporalMetrics.main()


if __name__ == "__main__":
    main()
