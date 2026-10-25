import torch
import numpy as np
import pandas as pd

from tabpfn.utils import meta_dataset_collator, fix_dtypes, process_text_na_dataframe
# from tabpfn.preprocessors.preprocessing_helpers import get_ordinal_encoder

# Otherwise, yoa-johnson double power can end up causing a lot of overflows...
DEFAULT_NUMPY_PREPROCESSING_DTYPE = np.float64

def preprocess_tensor(data: torch.Tensor, cat_indices: list[int], cat_cardinalities: list[int]):
    """
    Preprocesses a tensor by one-hot encoding categorical values and 
    standardizing numerical ones.

    Parameters:
    - data (torch.Tensor): Shape (n_samples, n_features)
    - cat_indices (list[int]): Indices of categorical features
    - cat_cardinalities (list[int]): Number of unique values for each categorical feature

    Returns:
    - torch.Tensor: Processed tensor with one-hot categorical and standardized numerical features
    """
    n_samples, n_features = data.shape
    processed_features = []

    for i in range(n_features):
        if i in cat_indices:
            # Find index in cat_indices to get cardinality
            idx_in_cat = cat_indices.index(i)
            cardinality = cat_cardinalities[idx_in_cat]
            # One-hot encode
            one_hot = torch.nn.functional.one_hot(
                data[:, i].long().clip(0), num_classes=cardinality
            )
            processed_features.append(one_hot.float())
        else:
            # Standardize numerical features
            col = data[:, i]
            mean = col.mean()
            std = col.std(unbiased=False)
            standardized = (col - mean) / (std + 1e-8)
            processed_features.append(standardized.unsqueeze(1))

    # Concatenate all transformed features
    return torch.cat(processed_features, dim=1)

def _get_ordinal_encoder(
    *,
    numpy_dtype: np.floating = DEFAULT_NUMPY_PREPROCESSING_DTYPE,  # type: ignore
) -> OrderPreservingColumnTransformer:
    """Create a ColumnTransformer that ordinally encodes string/category columns."""
    oe = OrdinalEncoder(
        # TODO: Could utilize the categorical dtype values directly instead of "auto"
        categories="auto",
        dtype=numpy_dtype,  # type: ignore
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,  # Missing stays missing
    )

    # Documentation of sklearn, deferring to pandas is misleading here. It's done
    # using a regex on the type of the column, and using `object`, `"object"` and
    # `np.object` will not pick up strings.
    to_convert = ["category", "string"]

    # Using a ColumnTransformer, where an inner transformer is applied only to a subset
    # of columns, does not retain the original column order of the data, which later
    # components in the PFN pipeline rely on (e.g., categorical indices).
    # Therefore, we use a custom class that, under certain constraints
    # (only OneToOneFeatureMixin transformers on disjoint column subsets),
    # reconstructs the original order after encoding.
    return OrderPreservingColumnTransformer(
        transformers=[("encoder", oe, make_column_selector(dtype_include=to_convert))],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

# def X_input_preprocessing(X: torch.Tensor, cat_idxs: list[int]):
#     X_numpy = X.cpu().numpy()
#     X_numpy = fix_dtypes(X_numpy, cat_indices=cat_idxs)

#     ord_encoder = get_ordinal_encoder()

#     X_preprocessed = process_text_na_dataframe(X_numpy, ord_encoder=ord_encoder, fit_encoder=True)
#     X_preprocessed = torch.as_tensor(X_preprocessed, dtype=torch.float32)

#     return X_preprocessed
