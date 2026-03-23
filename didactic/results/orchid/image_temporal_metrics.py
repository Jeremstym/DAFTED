from dataprocessing import get_vital_root

from didactic.results.orchid.utils.temporal_metrics import TemporalMetrics
from didactic.results.orchid.utils.time_series_attributes import TimeSeriesAttributesMixin


class ImageTemporalMetrics(TimeSeriesAttributesMixin, TemporalMetrics): # type: ignore[misc]
    """Class that computes temporal coherence metrics on image time-series attributes."""

    desc = f"seg_{TemporalMetrics.desc}"
    default_attribute_statistics_cfg = get_vital_root() / "data/camus/statistics/image_attr_stats.yaml"


def main():
    """Run the script."""
    ImageTemporalMetrics.main()


if __name__ == "__main__":
    main()
