import numpy as np
import pytest

from spatialshap import Explainer


def test_result_arrays_are_read_only_and_tabular_output_is_auditable():
    background = np.array([[0.0, 1.0], [2.0, 3.0]])
    explanation = Explainer(
        lambda values: np.asarray(values).sum(axis=1),
        background,
    )(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError):
        explanation.values[0, 0] = 999.0

    frame = explanation.to_frame()
    assert {"prediction", "base_value", "phi__feature_0", "phi__feature_1"} <= set(
        frame.columns
    )
    assert frame.loc[0, "additivity_error"] == pytest.approx(0.0)


def test_row_and_feature_slicing_preserve_dimensions():
    background = np.array([[0.0, 1.0], [2.0, 3.0]])
    explanation = Explainer(
        lambda values: np.asarray(values).sum(axis=1),
        background,
    )(np.array([[1.0, 2.0], [3.0, 4.0]]))

    one_row = explanation[0]
    assert one_row.shape == (1, 2)

    one_feature = explanation[:, "feature_1"]
    assert one_feature.shape == (2, 1)
    assert one_feature.feature_names == ("feature_1",)


def test_summary_diagnostics_save_and_slice_errors(tmp_path):
    background = np.array([[0.0, 1.0], [2.0, 3.0]])
    explanation = Explainer(
        lambda values: np.asarray(values).sum(axis=1),
        background,
    )(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert "SpatialExplanation" in explanation.summary()
    assert len(explanation.diagnostics()) == 2

    destination = tmp_path / "explanation.npz"
    explanation.save(str(destination))
    archive = np.load(destination)
    assert archive["values"].shape == (2, 2)

    with pytest.raises(KeyError):
        _ = explanation[:, "missing"]
    with pytest.raises(IndexError):
        _ = explanation[(slice(None), slice(None), slice(None))]
    with pytest.raises(ValueError, match="does not contain geometry"):
        explanation.to_geodataframe()
