import json
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def test_cosine_similarity_tool():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using build_dtm_tool from tools_dtm
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get cosine_similarity_tool
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 1. Known pairs verification (#16)
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 1})
    assert res_0_1["status"] == "success"
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-4)

    res_0_3 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 3})
    assert res_0_3["status"] == "success"
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-4)

    res_3_4 = cosine_similarity_tool.invoke({"doc_index_a": 3, "doc_index_b": 4})
    assert res_3_4["status"] == "success"
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-4)

    # 2. Empty text row 7 comparison (must be exactly 0.0)
    res_0_7 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 7})
    assert res_0_7["status"] == "success"
    assert res_0_7["cosine_similarity"] == 0.0

    # 3. Out-of-bounds index errors
    res_oob_pos = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 8})
    assert res_oob_pos["status"] == "error"

    res_oob_neg = cosine_similarity_tool.invoke({"doc_index_a": -1, "doc_index_b": 2})
    assert res_oob_neg["status"] == "error"

    # JSON serializability check on summary
    json_str = json.dumps(res_0_1)
    assert isinstance(json_str, str)

def test_cosine_similarity_no_dtm_error():
    # Fresh session without DTM
    session_no_dtm = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session_no_dtm)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    exploration_tools = make_exploration_tools(session_no_dtm)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    res = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 1})
    assert res["status"] == "error"

def test_feature_correlation_matrix_tool():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using build_dtm_tool from tools_dtm
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get feature_correlation_matrix_tool
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # Call tool
    res = feature_corr_tool.invoke({"top_n": 20})
    assert res["status"] == "success"

    # 1. Term order check by variance (top terms on sample_fixture)
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # 2. Check correlation values
    matrix = np.array(res["correlation_matrix"])
    idx = {term: i for i, term in enumerate(expected_terms)}

    # Diagonal elements all 1.0
    for i in range(len(expected_terms)):
        assert matrix[i, i] == pytest.approx(1.0, abs=1e-4)

    # Specific pairs from known answer #19
    assert matrix[idx["alpha"], idx["beta"]] == pytest.approx(0.8885, abs=1e-4)
    assert matrix[idx["gamma"], idx["delta"]] == pytest.approx(0.8307, abs=1e-4)
    assert matrix[idx["alpha"], idx["delta"]] == pytest.approx(-0.6882, abs=1e-4)

    # 3. Symmetry check and pending figure check
    assert np.allclose(matrix, matrix.T)
    assert session.pending_figure is not None

    # 5. JSON serializability check
    json_str = json.dumps(res)
    assert isinstance(json_str, str)

def test_feature_correlation_matrix_no_dtm_error():
    # 4. New session without DTM returns error
    session_no_dtm = SessionState()
    exploration_tools = make_exploration_tools(session_no_dtm)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({"top_n": 20})
    assert res["status"] == "error"
