import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_empty_filtering():
    """Test #17 part 1: variance and term_frequency filtering methods return empty results."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    for category in ["catA", "catB"]:
        for filtering in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": category,
                "filtering_method": filtering,
                "min_sup": 1
            })

            assert res["status"] == "success"
            assert res["patterns_count"] == 0
            assert res["top_patterns"] == []
            assert res["note"] == "no terms survived filtering"


def test_mine_patterns_tfidf_fpgrowth():
    """Test #17 part 2: tfidf filtering method with fpgrowth algorithm."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # catA min_sup=1
    res_catA_sup1 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA_sup1["status"] == "success"
    assert res_catA_sup1["patterns_count"] == 1
    assert res_catA_sup1["top_patterns"][0]["pattern"] == "alpha"
    assert res_catA_sup1["top_patterns"][0]["support"] == 3

    # catA min_sup=4
    res_catA_sup4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catA_sup4["status"] == "success"
    assert res_catA_sup4["patterns_count"] == 0
    assert res_catA_sup4["top_patterns"] == []

    # catB min_sup=1
    res_catB_sup1 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB_sup1["status"] == "success"
    assert res_catB_sup1["patterns_count"] == 1
    assert res_catB_sup1["top_patterns"][0]["pattern"] == "gamma"
    assert res_catB_sup1["top_patterns"][0]["support"] == 4

    # catB min_sup=4
    res_catB_sup4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catB_sup4["status"] == "success"
    assert res_catB_sup4["patterns_count"] == 1
    assert res_catB_sup4["top_patterns"][0]["pattern"] == "gamma"
    assert res_catB_sup4["top_patterns"][0]["support"] == 4
