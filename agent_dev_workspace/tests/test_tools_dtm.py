import json
import pytest
from pytest import approx
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool():
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture CSV
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Get build_dtm_tool from student's tools_dtm module
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    # 1 & 2. Test standard execution matching known answer #4
    res = build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})
    
    assert res["status"] == "success"
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == approx(47.5, abs=1e-3)
    
    # Check session state updates
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)

def test_build_dtm_tool_validation():
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    # 3. Validation errors
    res_invalid_range = build_dtm_tool.invoke({"ngram_range": [2, 1]})
    assert res_invalid_range["status"] == "error"
    
    res_invalid_length = build_dtm_tool.invoke({"ngram_range": [1]})
    assert res_invalid_length["status"] == "error"

def test_term_frequency_tool():
    # Test 1: Error when DTM has not been built yet
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    tf_tool_empty = next(t for t in tools_empty if t.name == "term_frequency_tool")
    
    res_no_dtm = tf_tool_empty.invoke({"top_n": 20})
    assert res_no_dtm["status"] == "error"
    assert "Document-Term Matrix has not been built yet" in res_no_dtm["message"]

    # Test 2: Build DTM using sample_fixture.csv
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    tools = make_tools(session)
    build_dtm = next(t for t in tools if t.name == "build_dtm_tool")
    term_freq = next(t for t in tools if t.name == "term_frequency_tool")

    dtm_res = build_dtm.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})
    assert dtm_res["status"] == "success"

    # Test 3: Invalid top_n (top_n == 0)
    res_top_n_zero = term_freq.invoke({"top_n": 0})
    assert res_top_n_zero["status"] == "error"
    assert "top_n must be a positive integer" in res_top_n_zero["message"]

    # Test 4: Known answer #14 check on sample_fixture.csv (top_n = 20)
    tf_res = term_freq.invoke({"top_n": 20})
    assert tf_res["status"] == "success"
    assert tf_res["total_terms"] == 5
    assert tf_res["top_n"] == 20

    top_terms = tf_res["top_terms"]
    assert len(top_terms) == 5

    # Check known frequencies: alpha=6, always=7, beta=3, delta=4, gamma=7
    assert top_terms["alpha"] == 6
    assert top_terms["always"] == 7
    assert top_terms["beta"] == 3
    assert top_terms["delta"] == 4
    assert top_terms["gamma"] == 7

    # Test 5: Verify top_n = 2 slicing works correctly
    tf_res_top2 = term_freq.invoke({"top_n": 2})
    assert tf_res_top2["status"] == "success"
    assert tf_res_top2["top_n"] == 2
    top2_terms = tf_res_top2["top_terms"]
    assert len(top2_terms) == 2
    assert list(top2_terms.keys()) == ["always", "gamma"]
    assert top2_terms["always"] == 7
    assert top2_terms["gamma"] == 7

    # Test 6: Verify JSON serializability of summary dict
    dumped_json = json.dumps(tf_res)
    assert isinstance(dumped_json, str)

    # Test 7: Verify stored full_report structure, column names, and sorting (including tie-breaking)
    stored_result = session.results_store[tf_res["result_id"]]
    full_report = stored_result["full_report"]
    
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "frequency"]

    # Expected order: always (7) -> gamma (7) -> alpha (6) -> delta (4) -> beta (3)
    expected_terms = ["always", "gamma", "alpha", "delta", "beta"]
    expected_freqs = [7, 7, 6, 4, 3]

    assert full_report["term"].tolist() == expected_terms
    assert full_report["frequency"].tolist() == expected_freqs

def test_dtm_heatmap_tool():
    # Test 1: Error when DTM/Dataset has not been built/loaded yet
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    dtm_heatmap_empty = next(t for t in tools_empty if t.name == "dtm_heatmap_tool")

    res_no_dtm = dtm_heatmap_empty.invoke({"n_terms": 20, "n_documents": 20})
    assert res_no_dtm["status"] == "error"
    assert "Document-Term Matrix or dataset has not been loaded/built" in res_no_dtm["message"]

    # Test 2: Build DTM using sample_fixture.csv
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    tools = make_tools(session)
    build_dtm = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap = next(t for t in tools if t.name == "dtm_heatmap_tool")

    dtm_res = build_dtm.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})
    assert dtm_res["status"] == "success"

    # Test 3: Invalid parameters (n_terms <= 0 or n_documents <= 0)
    res_invalid_terms = dtm_heatmap.invoke({"n_terms": 0, "n_documents": 20})
    assert res_invalid_terms["status"] == "error"
    assert "positive integers" in res_invalid_terms["message"]

    res_invalid_docs = dtm_heatmap.invoke({"n_terms": 20, "n_documents": -1})
    assert res_invalid_docs["status"] == "error"
    assert "positive integers" in res_invalid_docs["message"]

    # Test 4: Default parameters (n_terms=20, n_documents=20) -> Known Answer #21
    res_default = dtm_heatmap.invoke({"n_terms": 20, "n_documents": 20})
    assert res_default["status"] == "success"
    assert res_default["n_documents_sliced"] == 8
    assert res_default["n_terms_sliced"] == 5
    assert res_default["term_labels"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_default["doc_labels"] == list(range(8))

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert res_default["matrix_slice"] == expected_full_matrix

    # Check pending_figure is set
    assert session.pending_figure is not None

    # Check json.dumps serializability
    dumped_json = json.dumps(res_default)
    assert isinstance(dumped_json, str)

    # Test 5: Slicing test with n_terms=2, n_documents=3
    res_sliced = dtm_heatmap.invoke({"n_terms": 2, "n_documents": 3})
    assert res_sliced["status"] == "success"
    assert res_sliced["n_documents_sliced"] == 3
    assert res_sliced["n_terms_sliced"] == 2
    assert res_sliced["term_labels"] == ["alpha", "always"]
    assert res_sliced["doc_labels"] == [0, 1, 2]
    assert res_sliced["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
