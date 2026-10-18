import json
import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool():
    # 1. Test error when DTM is not built yet
    fresh_session = SessionState()
    filtering_tools_fresh = make_filtering_tools(fresh_session)
    var_tool_fresh = next(t for t in filtering_tools_fresh if t.name == "variance_filter_tool")
    
    err_res_no_dtm = var_tool_fresh.invoke({"threshold": 0.0})
    assert err_res_no_dtm["status"] == "error"
    assert "Document-Term Matrix has not been built" in err_res_no_dtm["message"]

    # 2. Setup session with dataset loaded and DTM built
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    var_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 3. Test invalid threshold (< 0.0)
    err_res_negative = var_tool.invoke({"threshold": -1.0})
    assert err_res_negative["status"] == "error"
    assert "threshold must be non-negative" in err_res_negative["message"]

    # 4. Test threshold = 0.15 on sample_fixture.csv (Known answer #5)
    res = var_tool.invoke({"threshold": 0.15})
    assert res["status"] == "success"

    # Verify JSON serializability
    json_str = json.dumps(res)
    assert isinstance(json_str, str)

    # Verify summary contents
    assert res["threshold"] == 0.15
    assert res["n_terms_kept"] == 4
    assert res["n_terms_removed"] == 1
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert res["removed_terms"] == ["always"]

    # Verify full_report stored in session
    res_id = res["result_id"]
    stored_entry = session.results_store[res_id]
    full_report = stored_entry["full_report"]
    
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "variance"]
    
    # Check exact sorted order in full_report: alpha -> gamma -> delta -> beta -> always
    expected_order = ["alpha", "gamma", "delta", "beta", "always"]
    assert list(full_report["term"]) == expected_order

    # Check variance values against TEST_FIXTURE.md #5
    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375
    }

    report_dict = dict(zip(full_report["term"], full_report["variance"]))
    for term, exp_var in expected_variances.items():
        assert report_dict[term] == pytest.approx(exp_var, abs=1e-4)

    # Verify session.feature_matrix was NOT modified
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)


def test_pearson_filter_tool():
    # 1. Test error when DTM is not built yet in a fresh session
    fresh_session = SessionState()
    filtering_tools_fresh = make_filtering_tools(fresh_session)
    pearson_tool_fresh = next(t for t in filtering_tools_fresh if t.name == "pearson_filter_tool")

    err_res_no_dtm = pearson_tool_fresh.invoke({"target_class": "catB"})
    assert err_res_no_dtm["status"] == "error"
    assert "Document-Term Matrix has not been built" in err_res_no_dtm["message"]

    # 2. Setup session with dataset loaded and DTM built
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # 3. Test error when target_class is not in categories
    err_res_invalid_class = pearson_tool.invoke({"target_class": "catC"})
    assert err_res_invalid_class["status"] == "error"
    assert "target_class 'catC' not found" in err_res_invalid_class["message"]

    # 4. Test target_class="catB" on sample_fixture.csv (Known answer #6)
    res = pearson_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"

    # Verify JSON serializability of summary (no numpy type issues)
    json_str = json.dumps(res)
    assert isinstance(json_str, str)

    # Verify summary details
    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5
    assert len(res["top_10_terms"]) == 5

    # Verify full_report stored in session
    res_id = res["result_id"]
    stored_entry = session.results_store[res_id]
    full_report = stored_entry["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "pearson_r"]

    # Check order by abs(r) descending: delta -> gamma -> beta -> alpha -> always
    expected_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert list(full_report["term"]) == expected_order

    # Check Pearson r values against TEST_FIXTURE.md #6 with abs=1e-4 tolerance
    expected_r = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307
    }

    report_dict = dict(zip(full_report["term"], full_report["pearson_r"]))
    for term, exp_val in expected_r.items():
        assert report_dict[term] == pytest.approx(exp_val, abs=1e-4)


def test_spearman_filter_tool():
    # 1. Test error when DTM is not built yet
    fresh_session = SessionState()
    filtering_tools_fresh = make_filtering_tools(fresh_session)
    spearman_tool_fresh = next(t for t in filtering_tools_fresh if t.name == "spearman_filter_tool")

    err_res_no_dtm = spearman_tool_fresh.invoke({"target_class": "catB"})
    assert err_res_no_dtm["status"] == "error"
    assert "Document-Term Matrix has not been built" in err_res_no_dtm["message"]

    # 2. Setup session with dataset loaded and DTM built
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 3. Test error cases
    err_res_invalid_class = spearman_tool.invoke({"target_class": "catC"})
    assert err_res_invalid_class["status"] == "error"
    assert "target_class 'catC' not found" in err_res_invalid_class["message"]

    err_res_invalid_max_terms = spearman_tool.invoke({"target_class": "catB", "max_terms": 0})
    assert err_res_invalid_max_terms["status"] == "error"
    assert "max_terms must be positive" in err_res_invalid_max_terms["message"]

    # 4. Test target_class="catB" with default max_terms (1500) (Known answer #6)
    res = spearman_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"

    # Verify json.dumps works on summary
    json_str = json.dumps(res)
    assert isinstance(json_str, str)

    # Verify summary details
    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5
    assert len(res["top_10_terms"]) == 5

    # Verify full_report stored in session
    res_id = res["result_id"]
    stored_entry = session.results_store[res_id]
    full_report = stored_entry["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "spearman_r"]

    # Verify exact column order sorted by abs(r) descending: delta -> gamma -> beta -> alpha -> always
    expected_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert list(full_report["term"]) == expected_order

    # Verify exact Spearman r values against TEST_FIXTURE.md #6 with abs=1e-4 tolerance
    expected_r = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363
    }

    report_dict = dict(zip(full_report["term"], full_report["spearman_r"]))
    for term, exp_val in expected_r.items():
        assert report_dict[term] == pytest.approx(exp_val, abs=1e-4)

    # 5. Test max_terms=2 filtering (should keep top 2 variance terms: alpha and gamma)
    res_max2 = spearman_tool.invoke({"target_class": "catB", "max_terms": 2})
    assert res_max2["status"] == "success"
    assert res_max2["total_terms"] == 2

    res_id_max2 = res_max2["result_id"]
    full_report_max2 = session.results_store[res_id_max2]["full_report"]
    assert set(full_report_max2["term"]) == {"alpha", "gamma"}
