import functools
import itertools
from typing import Any, Callable, Dict, Literal, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
import torchdata.datapipes as dp
from dataprocessing.data.orchid.config import OrchidTag, TabularAttribute, TimeSeriesAttribute
from dataprocessing.data.orchid.config import View as ViewEnum
from dataprocessing.data.orchid.utils.attributes import TABULAR_CAT_ATTR_LABELS
from dataprocessing.data.orchid.utils.data_struct import MISSING_TS_ATTR, MISSING_TS_VIEWS, Patient
from dataprocessing.data.orchid.utils.itertools import Patients
from torch import Tensor
from torch.utils.data import MapDataPipe
from torchvision.transforms import transforms

T = TypeVar("T", np.ndarray, Tensor)
PatientData = Dict[
    Union[str, TabularAttribute, ViewEnum], Union[Patient.Id, T, Dict[Union[TimeSeriesAttribute, OrchidTag], T]]
]
PatientDataTarget = Dict[
    Union[str, TabularAttribute, ViewEnum], T
]
PatientDataInference = Dict[
    Union[str, TabularAttribute, ViewEnum], T
]
AttributeTransform = Callable[[np.ndarray], np.ndarray]

MISSING_NUM_ATTR = np.nan
MISSING_CAT_ATTR = -1
# MISSING_TS_VIEWS = {view_enum: {attr_tag: MISSING_NUM_ATTR for attr_tag in TimeSeriesAttribute} for view_enum in ViewEnum}


def dicts_equal(di1, di2):
    if set(di1.keys()) != set(di2.keys()):
        return False
    for k in di1:
        v1, v2 = di1[k], di2[k]
        # Use np.array_equal for exact, or np.allclose for approximate equality
        if not np.array_equal(v1, v2):
            return False
    return True


def build_datapipes(
    patients: Patients,
    process_patient_kwargs: Dict[str, Any] = None,
    transform_patient_kwargs: Dict[str, Any] = None,
    cache: bool = True,
) -> MapDataPipe[PatientData]:
    """Builds a pipeline of datapipes for the orchid dataset.

    Args:
        patients: Collection of patients to process in the pipeline of datapipes.
        process_patient_kwargs: Parameter to forward to the `process_patient` function.
        transform_patient_kwargs: Parameter to forward to the `transform_patient_data` function.
        cache: Whether to store elements at the end of the pipeline in memory (to avoid repetitive I/O and processing).

    Returns:
        Pipeline of datapipes for the orchid dataset.
    """
    if process_patient_kwargs is None:
        process_patient_kwargs = {}
    if transform_patient_kwargs is None:
        transform_patient_kwargs = {}

    datapipe = dp.map.SequenceWrapper(list(patients.keys()))
    datapipe = datapipe.map(patients.__getitem__)  # Load patient from disk
    datapipe = datapipe.map(functools.partial(process_patient, **process_patient_kwargs))
    if cache:
        # Since the orchid is relatively small, it can be cached in memory to avoid repetitive I/O
        datapipe = datapipe.in_memory_cache()
    datapipe = datapipe.map(functools.partial(transform_patient_data, **transform_patient_kwargs))
    return datapipe


def process_patient(
    patient: Patient,
    tabular_attrs: Sequence[TabularAttribute] = None,
    time_series_attrs: Sequence[TimeSeriesAttribute] = None,
    mask_tag: str = OrchidTag.mask,
    bmode_tag: str = None,
) -> PatientData:
    """Processes a patient to extract and format a subset of relevant fields from all the patient's data.

    Args:
        patient: Patient to process.
        tabular_attrs: Tabular attributes to extract from the patient.
        time_series_attrs: Time-series attributes to extract from the patient.
        mask_tag: Tag of the segmentation mask to use for computing the time-series attrs, and to include in the item.
        bmode_tag: Tag of the B-mode image to which the segmentation mask applies. Providing it causes the returned item
            to include image data.

    Returns:
        Each item returned contains two types of attributes: time-series attributes and tabular attributes.
        - Time-series attributes are 1D vectors w.r.t. time (one value per frame per view).
        - Tabular attributes are scalars (one value overall for the patient).
        If a `bmode_tag` value was provided, the items will also contain the B-mode and mask sequences from the patient.
    """
    if tabular_attrs is None:
        tabular_attrs = TabularAttribute
    if time_series_attrs is None:
        time_series_attrs = TimeSeriesAttribute
    img_data_tags = [bmode_tag, mask_tag] if bmode_tag else []

    if tabular_attrs:
        # Extract the requested tabular attributes from all the ones available for the patient,
        tab_attrs_data = {attr_tag: patient.attrs.get(attr_tag) for attr_tag in tabular_attrs}

        for attr_tag, attr in tab_attrs_data.items():
            if attr_tag in TabularAttribute.numerical_attrs():
                # Convert numerical attributes to numpy arrays of dtype `np.float32`
                try:
                    tab_attrs_data[attr_tag] = np.array(
                        attr if attr is not None else MISSING_NUM_ATTR, dtype=np.float32
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Tabular attribute {attr_tag} of patient {patient.id} raise the issue {e} for value {attr}"
                    ) from e
            else:
                # Convert categorical attributes to numerical labels inside numpy arrays of dtype `np.int64`
                try:
                    tab_attrs_data[attr_tag] = np.array(
                        TABULAR_CAT_ATTR_LABELS[attr_tag].index(attr) if attr is not None else MISSING_CAT_ATTR,
                        dtype=np.int64,
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Tabular attribute {attr_tag} of patient {patient.id} raise the issue {e} for value {attr}"
                    ) from e
    else:
        tab_attrs_data = {}

    if time_series_attrs:
        # Prepare attributes computed from the images
        time_series_attrs_data = patient.get_mask_attributes(mask_tag)
        # Make sure the attributes array are of dtype `np.float32`, so that they'll be converted to dtype `torch.float`
        time_series_attrs_data = {
            view_enum: {
                attr_tag: np.array(attr, dtype=np.float32)  # Ensure attr is a NumPy array
                for attr_tag, attr in view_data.items()
                if attr_tag in time_series_attrs
            }
            if not np.isnan(list(view_data.values())[0]).all()
            else MISSING_TS_VIEWS
            # if not dicts_equal(view_data, MISSING_TS_VIEWS)
            for view_enum, view_data in time_series_attrs_data.items()
        }
    else:
        time_series_attrs_data = {}

    item = {"id": patient.id, **tab_attrs_data, **time_series_attrs_data}

    # Add the image data from the types of data listed in `self.img_data_tags`
    for img_data_tag, (view_enum, view) in itertools.product(img_data_tags, patient.views.items()):
        # In case we have images for views for which we did not have attributes, make sure to create a
        # dictionary for the view's data
        item.setdefault(view_enum, {})[img_data_tag] = view.data[img_data_tag]

    return item


def transform_patient_data(
    patient_data: PatientData,
    tabular_attrs_transforms: Dict[Union[Literal["any"], TabularAttribute], AttributeTransform] = None,
    time_series_attrs_transforms: Dict[Union[Literal["any"], TimeSeriesAttribute], AttributeTransform] = None,
) -> PatientData:
    """Applies transformations to patient, with generic funcs applied to modalities or attribute-specific funcs.

    Args:
        patient_data: Patient data.
        tabular_attrs_transforms: Mapping between tabular attributes and the transformations to apply to them.
            Transformations under the `any` key will be applied to all the attributes.
        time_series_attrs_transforms: Mapping between time-series attributes and the transformations to apply to them.
            Transformations under the `any` key will be applied to all the attributes.

    Returns:
        Patient data, where the requested tabular and time-series attributes where transformed.
    """
    if tabular_attrs_transforms is None:
        tabular_attrs_transforms = {}
    if time_series_attrs_transforms is None:
        time_series_attrs_transforms = {}

    tab_attrs = [key for key in patient_data if key in list(TabularAttribute)]
    views = [key for key in patient_data if key in list(ViewEnum)]

    # Apply tabular attributes transforms
    for tab_attr in tab_attrs:
        # Store the original dtype of the attribute
        attr_dtype = patient_data[tab_attr].dtype

        # If a generic tabular attribute transform is provided, apply it first
        if transform := tabular_attrs_transforms.get("any"):
            patient_data[tab_attr] = transform(patient_data[tab_attr])

        # If a transformation targets the current attribute specifically, apply it
        if transform := tabular_attrs_transforms.get(tab_attr):
            patient_data[tab_attr] = transform(patient_data[tab_attr])

        # Make sure that the dtype of the attribute was not modified by the transforms
        patient_data[tab_attr] = patient_data[tab_attr].astype(attr_dtype)

    # Apply time-series attributes transforms
    for view_data in (patient_data[view] for view in views):
        for time_series_attr in view_data:
            # Check if the time-series attribute is missing
            attr = view_data[time_series_attr]
            # if attr is None or np.isnan(attr).all() or np.array_equal(attr, MISSING_NUM_ATTR):
            #     continue  # Skip transformation for missing attributes

            # Store the original dtype of the attribute
            attr_dtype = attr.dtype

            # If a generic time-series attribute transform is provided, apply it first
            if transform := time_series_attrs_transforms.get("any"):
                view_data[time_series_attr] = transform(attr)

            # If a transformation targets the current attribute specifically, apply it
            if transform := time_series_attrs_transforms.get(time_series_attr):
                view_data[time_series_attr] = transform(view_data[time_series_attr])

            # Make sure that the dtype of the attribute was not modified by the transforms
            view_data[time_series_attr] = view_data[time_series_attr].astype(attr_dtype)

    return patient_data


def filter_time_series_attributes(
    item_or_batch: PatientData,
    views: Sequence[ViewEnum] = tuple(ViewEnum),
    attrs: Sequence[TimeSeriesAttribute] = tuple(TimeSeriesAttribute),
) -> Tuple[Dict[Tuple[ViewEnum, TimeSeriesAttribute], T], Tensor]:
    """Filters an item/batch of data from a `PatientProcessor` to only keep requested time-series attrs from some views.

    Args:
        item_or_batch: Item/batch of data from a `PatientProcessor`.
        views: Views for which to keep time-series attributes data.
        attrs: Attributes for which to keep data.

    Returns:
        Requested time-series attributes from the requested views in the item/batch of data.
    """
    time_series_data = {
        (view_enum, view_data_tag): data
        for view_enum in views
        for view_data_tag, data in item_or_batch.get(view_enum, {}).items()
        if view_data_tag in attrs
    }
    if time_series_data:
        time_series_values = torch.stack(list(time_series_data.values()), dim=1)
        if time_series_values.ndim == 1:
            raise ValueError(f"Time-series data is 1D, but expected 2D or 3D. Data: {time_series_values}")
        elif time_series_values.ndim == 2:
            # If the data is 2D, add a dimension to make it 3D (add batch dimension)
            # time_series_values = time_series_values.unsqueeze(0).permute(0, 2, 1)
            time_series_notna_mask = torch.all(~torch.isnan(time_series_values), dim=0)
        elif time_series_values.ndim == 3:
            time_series_notna_mask = torch.all(~torch.isnan(time_series_values), dim=2)
        else:
            raise ValueError(
                f"Time-series data is {time_series_values.ndim}D, but expected 2D or 3D. Data: {time_series_values}"
            )
    else:
        time_series_notna_mask = None

    # time_series_notna_mask = torch.stack(
    #     [
    #         # not np.array_equal(data, MISSING_TS_VIEWS)
    #         ~(torch.isnan(data).all()) if data.ndim == 1 else torch.all(torch.isnan(data), dim=1)
    #         for view_enum, view_data_tag in time_series_data
    #         for data in time_series_data[(view_enum, view_data_tag)]
    #     ],
    #     # dim=1
    # ).reshape(-1, len(attrs) * len(views))
    # print(f'time_series_notna_mask: {time_series_notna_mask.shape}')
    # time_series_notna_mask = list(
    #     map(
    #         lambda data: not torch.isnan(data).all(),
    #         itertools.chain.from_iterable(
    #             [time_series_data[(view_enum, view_data_tag)] for view_enum, view_data_tag in time_series_data]
    #         ),
    #     )
    # )
    # time_series_notna_mask = torch.Tensor(
    #     [
    #         not torch.isnan(data).all()
    #         for data in time_series_data.values()
    #     ]
    # ).reshape(-1, len(attrs) * len(views))

    # Check if mask match torch.nan data
    # for (view_enum, view_data_tag), mask in zip(time_series_data, time_series_notna_mask):
    #     data = time_series_data[(view_enum, view_data_tag)]
    #     if mask and torch.isnan(data).any():
    #         # If the mask is True, but the data contains NaN values, raise an error
    #         raise ValueError(
    #             f"Mask does not match torch.nan data for {view_enum}/{view_data_tag}: {data[~torch.isnan(data)]} != {data[mask]}"
    #         )
    #     elif not mask and not torch.isnan(data).any():
    #         # If the mask is False, but the data does not contain NaN values, raise an error
    #         raise ValueError(
    #             f"Mask does not match torch.nan data for {view_enum}/{view_data_tag}: {data[~torch.isnan(data)]} != {data[mask]}"
    #         )

    return time_series_data, time_series_notna_mask


if __name__ == "__main__":
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt
    from dataprocessing.data.orchid.config import Label
    from dataprocessing.data.orchid.utils.attributes import build_attributes_dataframe, plot_attributes_wrt_time
    from dataprocessing.utils.importlib import import_from_module
    from dataprocessing.utils.parsing import yaml_flow_collection
    from PIL import Image
    from skimage import color

    parser = ArgumentParser()
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=OrchidTag.mask,
        help="Tag of the segmentation mask to use for computing the time-series attributes, and to include in the item "
        "if `include_img_data=True`",
    )
    parser.add_argument(
        "--bmode_tag",
        type=str,
        help="Tag of the B-mode image to which the segmentation mask applies, only used if image data is to be "
        "included (`include_img_data=True`)",
    )
    parser.add_argument(
        "--time_series_attrs_transforms",
        type=yaml_flow_collection,
        default={},
        metavar="{XFORM1_CLASSPATH:{ARG1:VAL1,...},...}",
        help="Transformations to apply to the time-series extracted from the images",
    )
    parser.add_argument(
        "--debug_transforms",
        action="store_true",
        help="Whether to plot the original data alongside the transformed data to visualize the transformations' "
        "impact",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    mask_tag, bmode_tag, time_series_attrs_transforms, debug_transforms = (
        kwargs.pop("mask_tag"),
        kwargs.pop("bmode_tag"),
        kwargs.pop("time_series_attrs_transforms"),
        kwargs.pop("debug_transforms"),
    )

    # Automatically build transforms from module paths
    time_series_attrs_transforms = {
        "any": transforms.Compose(
            [
                import_from_module(transform_classpath)(**transform_kwargs)
                for transform_classpath, transform_kwargs in time_series_attrs_transforms.items()
            ]
        )
    }

    # Create the patients' datapipe
    patients = Patients(**kwargs)
    process_patient_kwargs = {"mask_tag": mask_tag, "bmode_tag": bmode_tag}
    datapipe = build_datapipes(
        patients,
        process_patient_kwargs=process_patient_kwargs,
        transform_patient_kwargs={"time_series_attrs_transforms": time_series_attrs_transforms},
    )

    # Draw the first patient from the datapipe
    first_item = datapipe[0]
    patient_id = list(patients.keys())[0]

    if debug_transforms:
        # Create another datapipe that doesn't apply transforms to gather untransformed data
        datapipe_no_xform = build_datapipes(patients, process_patient_kwargs=process_patient_kwargs)
        first_item_no_xform = datapipe_no_xform[0]

    for view, data in ((view, first_item[view]) for view in ViewEnum if view in first_item):
        # Plot image data
        if bmode_tag:
            overlaid_img = np.array(
                [
                    color.label2rgb(mask, image=im, bg_label=Label.BG)
                    for mask, im in zip(data[args.mask_tag], data[args.bmode_tag])
                ]
            )
            overlaid_img = (overlaid_img * 255).astype(np.uint8)
            overlaid_frames = [Image.fromarray(overlaid_frame) for overlaid_frame in overlaid_img]
            # TODO Display image data interactively as an animated GIF
            # overlaid_frames[0].show(f"{patient_id}/{view_enum} {args.bmode_tag}/{args.mask_tag}")

        attrs = {}

        # Start by collecting untransformed data so that it will be plotted first in the graph (more natural plot)
        if debug_transforms:
            # If debugging transforms, also extract time-series attributes from the original item
            no_xform_view_data = first_item_no_xform[view]
            attrs["og"] = {
                attr_tag: attr for attr_tag, attr in no_xform_view_data.items() if attr_tag in list(TimeSeriesAttribute)
            }

        # Extract the time-series attributes from the item
        item_tag = mask_tag if not debug_transforms else "xform"
        attrs[item_tag] = {attr_tag: attr for attr_tag, attr in data.items() if attr_tag in list(TimeSeriesAttribute)}

        # Plot the curves for each attribute w.r.t. time
        for _ in plot_attributes_wrt_time(
            build_attributes_dataframe(attrs, normalize_time=True), plot_title_root=f"{patient_id}/{view}"
        ):
            plt.show()
            plt.close()
