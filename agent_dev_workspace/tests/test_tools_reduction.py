import json
import pytest
import pandas as pd
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_binarize_labels_tool_success():
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM to set session.labels and session.categories
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Invoke binarize_labels_tool
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    res = binarize_labels_tool.invoke({})
    
    # Assert return status and structure
    assert res["status"] == "success"
    assert res["shape"] == [8, 2]
    assert res["categories"] == ["catA", "catB"]
    assert res["counts"]["catA"] == 4
    assert res["counts"]["catB"] == 4
    
    # Check artifact DataFrame stored in session
    df_onehot = session.artifacts["binarized_labels"]
    assert isinstance(df_onehot, pd.DataFrame)
    assert df_onehot.shape == (8, 2)
    assert list(df_onehot.columns) == ["catA", "catB"]
    
    # Check specific row vectors
    assert list(df_onehot.iloc[0]) == [1, 0]
    assert list(df_onehot.iloc[3]) == [0, 1]
    assert list(df_onehot.iloc[7]) == [1, 0]
    
    # Check row and column sums
    assert (df_onehot.sum(axis=1) == 1).all()
    assert df_onehot["catA"].sum() == 4
    assert df_onehot["catB"].sum() == 4
    
    # Check JSON serializability of returned summary
    json_str = json.dumps(res)
    assert isinstance(json_str, str)


def test_binarize_labels_tool_no_labels_error():
    session = SessionState()
    
    # Load dataset only, WITHOUT building DTM so session.labels remains None
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Confirm labels is None
    assert session.labels is None
    
    # Invoke binarize_labels_tool and check error response
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    res = binarize_labels_tool.invoke({})
    
    assert res["status"] == "error"
    assert "message" in res


def test_reduce_dimensions_pca():
    session = SessionState()
    
    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM using tools_dtm
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    # Load reduction tools
    tools = make_reduction_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")
    
    # 1. Test error when perplexity >= n_samples (8 docs)
    err_res = reduce_tool.invoke({"method": "tsne", "perplexity": 10.0})
    assert err_res["status"] == "error"
    assert "perplexity" in err_res["message"]
    
    # 2. Test PCA execution (known answer #7)
    res = reduce_tool.invoke({"method": "pca", "n_components": 2})
    assert res["status"] == "success"
    
    # Check pending figure is created
    assert session.pending_figure is not None
    
    # Check JSON serializability of summary
    json_str = json.dumps(res)
    assert isinstance(json_str, str)
    
    # Check explained variance ratio ([0.7532, 0.1965])
    evr = res["explained_variance_ratio"]
    assert pytest.approx(evr[0], abs=1e-4) == 0.7532
    assert pytest.approx(evr[1], abs=1e-4) == 0.1965
    
    # Check all 8 document coordinates from full_report against known answer #7
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    
    # Retrieve stored full report
    stored_result = session.results_store[res["result_id"]]
    full_report_df = stored_result["full_report"]
    actual_coords = full_report_df[["Dim1", "Dim2"]].values
    
    assert np.allclose(actual_coords, expected_coords, atol=1e-4)


def test_reduce_dimensions_tsne():
    session = SessionState()
    
    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM using tools_dtm
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    # Load reduction tools
    tools = make_reduction_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")
    
    # Run 1: t-SNE execution with perplexity=2, random_state=42
    res1 = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })
    assert res1["status"] == "success"
    
    # 1. Check pending figure is created
    assert session.pending_figure is not None
    
    # Check JSON serializability of summary
    json_str = json.dumps(res1)
    assert isinstance(json_str, str)
    
    # Retrieve full_report coordinates from run 1
    stored_result1 = session.results_store[res1["result_id"]]
    coords1 = stored_result1["full_report"][["Dim1", "Dim2"]].values
    
    # 2. Check output shape is (8, 2) and all values are finite
    assert coords1.shape == (8, 2)
    assert np.isfinite(coords1).all()
    
    # 3. Check duplicate documents (doc 5 and doc 6) have identical coordinates
    assert np.allclose(coords1[5], coords1[6], atol=1e-5)
    
    # 4. Run 2: Repeat with same random_state=42 and verify reproducibility
    res2 = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })
    stored_result2 = session.results_store[res2["result_id"]]
    coords2 = stored_result2["full_report"][["Dim1", "Dim2"]].values
    
    assert np.allclose(coords1, coords2, atol=1e-5)


def test_reduce_dimensions_umap():
    session = SessionState()
    
    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM using tools_dtm
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    # Load reduction tools
    tools = make_reduction_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")
    
    # Test UMAP execution (known answer #9)
    res = reduce_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    assert res["status"] == "success"
    
    # Check pending figure is created
    assert session.pending_figure is not None
    
    # Check JSON serializability of summary
    json_str = json.dumps(res)
    assert isinstance(json_str, str)
    
    # Check all 8 document coordinates from full_report against known answer #9 table
    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])
    
    # Retrieve stored full report
    stored_result = session.results_store[res["result_id"]]
    full_report_df = stored_result["full_report"]
    actual_coords = full_report_df[["Dim1", "Dim2"]].values
    
    assert np.allclose(actual_coords, expected_coords, atol=1e-4)
