import os
import json
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_check_missing_tool_fixture():
    # 1. Create a fresh session instance
    session = SessionState()
    
    # 2. Load premade tools to get load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 3. Load sample_fixture.csv into session
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    # 4. Instantiate tools from student's tools_data.py using the SAME session
    student_tools = make_tools(session)
    check_missing_tool = next(t for t in student_tools if t.name == "check_missing_tool")
    
    # 5. Invoke check_missing_tool
    result = check_missing_tool.invoke({"text_column": "text"})
    
    # 6. Assert against Known Answer #2 from TEST_FIXTURE.md:
    # - Exactly 8 total documents
    # - Exactly 1 missing row (index 7)
    # - missing_pct is 12.5% (1/8)
    assert result["status"] == "success"
    assert result["total_documents"] == 8
    assert result["missing_count"] == 1
    assert result["missing_pct"] == pytest.approx(12.5, abs=1e-2)
    assert result["sample_missing_indices"] == [7]

def test_check_duplicates_tool_detect_fixture():
    # Fresh session for duplicate detection test
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    student_tools = make_tools(session)
    check_duplicates_tool = next(t for t in student_tools if t.name == "check_duplicates_tool")
    
    # Test duplicate detection without dropping (drop=False)
    result = check_duplicates_tool.invoke({"drop": False})
    
    # Known Answer #3: Exactly 1 duplicate (row index 6, copy of row index 5)
    assert result["status"] == "success"
    assert result["total_documents"] == 8
    assert result["duplicate_count"] == 1
    assert result["sample_duplicate_indices"] == [6]
    assert result["dropped"] is False
    assert result["remaining_documents"] == 8

def test_check_duplicates_tool_drop_fixture():
    # Fresh session for duplicate dropping test to isolate session mutation
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    student_tools = make_tools(session)
    check_duplicates_tool = next(t for t in student_tools if t.name == "check_duplicates_tool")
    
    # Test duplicate dropping with keep=False (drop=True)
    result = check_duplicates_tool.invoke({"drop": True})
    
    # Known Answer #3: Both row 5 and row 6 ("always gamma delta") are dropped
    assert result["status"] == "success"
    assert result["dropped"] is True
    assert result["remaining_documents"] == 6
    assert len(session.dataframe) == 6
    
    # Specific assertion: verify "always gamma delta" no longer exists in session.dataframe["text"]
    assert "always gamma delta" not in session.dataframe["text"].values

def test_list_files_tool_known_answer_and_security():
    session = SessionState()
    student_tools = make_tools(session)
    list_files_tool = next(t for t in student_tools if t.name == "list_files_tool")
    
    # 1. Check Known Answer #11: subdirectory="newdataset"
    res_known = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res_known["status"] == "success"
    assert res_known["directories"] == []
    assert res_known["files"] == ["Reddit-stock-sentiment.csv"]
    
    # 2. Check security block for hidden directory: subdirectory=".git"
    res_git = list_files_tool.invoke({"subdirectory": ".git"})
    assert res_git["status"] == "error"
    assert "Access denied" in res_git["message"]
    
    # 3. Check security block for directory traversal outside root: subdirectory="../../"
    res_traversal = list_files_tool.invoke({"subdirectory": "../../"})
    assert res_traversal["status"] == "error"
    assert "Access denied" in res_traversal["message"]

def test_sample_data_tool_known_answer_13():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    student_tools = make_tools(session)
    sample_data_tool = next(t for t in student_tools if t.name == "sample_data_tool")
    
    # Known Answer #13: n=4, random_state=42
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})
    
    # Assert return summary matching Known Answer #13
    assert result["status"] == "success"
    assert result["original_count"] == 8
    assert result["sampled_count"] == 4
    assert result["sample_indices"] == [1, 5, 0, 7]
    assert result["sample_indices_count"] == 4
    
    # Assert session.dataframe was updated and reset to contiguous indices 0..3
    assert session.dataframe is not None
    assert len(session.dataframe) == 4
    assert list(session.dataframe.index) == [0, 1, 2, 3]

def test_sample_data_tool_missing_args_error():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    student_tools = make_tools(session)
    sample_data_tool = next(t for t in student_tools if t.name == "sample_data_tool")
    
    # Edge case 1: both n and frac are None -> returns error
    res_none = sample_data_tool.invoke({})
    assert res_none["status"] == "error"
    assert "Please specify either 'n'" in res_none["message"]

def test_sample_data_tool_both_args_error():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    student_tools = make_tools(session)
    sample_data_tool = next(t for t in student_tools if t.name == "sample_data_tool")
    
    # Edge case 2: both n and frac provided -> returns error
    res_both = sample_data_tool.invoke({"n": 4, "frac": 0.5})
    assert res_both["status"] == "error"
    assert "not both" in res_both["message"]

def test_sample_data_tool_oversized_n():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    student_tools = make_tools(session)
    sample_data_tool = next(t for t in student_tools if t.name == "sample_data_tool")
    
    # Edge case 3: n exceeds total documents (n=10 on 8-row dataset) -> capped with note
    res_oversized = sample_data_tool.invoke({"n": 10, "random_state": 42})
    assert res_oversized["status"] == "success"
    assert res_oversized["sampled_count"] == 8
    assert "note" in res_oversized
    assert "Capped sample size to 8" in res_oversized["note"]

def test_tokenize_tool_known_answer_10():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    student_tools = make_tools(session)
    tokenize_tool = next(t for t in student_tools if t.name == "tokenize_tool")
    
    # Invoke tokenize_tool with defaults
    result = tokenize_tool.invoke({"text_column": "text", "lowercase": True})
    
    # Assert summary stats matching Known Answer #10
    assert result["status"] == "success"
    assert result["total_documents"] == 8
    assert result["total_tokens"] == 27
    assert result["vocabulary_size"] == 5
    
    # Assert "unigrams" column is present in session.dataframe
    assert "unigrams" in session.dataframe.columns
    
    # Assert tokens for all 8 documents against Known Answer #10 table
    expected_tokens_per_doc = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    assert list(session.dataframe["unigrams"]) == expected_tokens_per_doc

def test_inspect_data_tool_designed_cases():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    student_tools = make_tools(session)
    inspect_data_tool = next(t for t in student_tools if t.name == "inspect_data_tool")
    
    # Case 1: position="first", n=2
    res_first = inspect_data_tool.invoke({"position": "first", "n": 2})
    assert res_first["status"] == "success"
    # Assertion 3: total_documents is 8
    assert res_first["total_documents"] == 8
    
    catA_first_idx = [r["index"] for r in res_first["categories"]["catA"]["sample_rows"]]
    catB_first_idx = [r["index"] for r in res_first["categories"]["catB"]["sample_rows"]]
    # Assertion 1: catA index [0, 1], catB index [3, 4]
    assert catA_first_idx == [0, 1]
    assert catB_first_idx == [3, 4]
    
    # Assertion 4: missing_text_count is 1 for catA (row 7) and 0 for catB
    assert res_first["categories"]["catA"]["missing_text_count"] == 1
    assert res_first["categories"]["catB"]["missing_text_count"] == 0
    
    # Case 2: position="last", n=2
    res_last = inspect_data_tool.invoke({"position": "last", "n": 2})
    assert res_last["status"] == "success"
    catA_last_idx = [r["index"] for r in res_last["categories"]["catA"]["sample_rows"]]
    catB_last_idx = [r["index"] for r in res_last["categories"]["catB"]["sample_rows"]]
    # Assertion 2: catA index [2, 7], catB index [5, 6]
    assert catA_last_idx == [2, 7]
    assert catB_last_idx == [5, 6]
    
    # Case 5: When n=10, both categories have a note
    res_large = inspect_data_tool.invoke({"n": 10})
    assert res_large["status"] == "success"
    assert "note" in res_large["categories"]["catA"]
    assert "note" in res_large["categories"]["catB"]
    
    # Case 6: n=0 returns an error
    res_zero = inspect_data_tool.invoke({"n": 0})
    assert res_zero["status"] == "error"
    
    # Case 7: json.dumps(result) works without errors
    dumped_json = json.dumps(res_first)
    assert isinstance(dumped_json, str)

def test_describe_data_tool_known_answer_18():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    student_tools = make_tools(session)
    describe_data_tool = next(t for t in student_tools if t.name == "describe_data_tool")
    
    result = describe_data_tool.invoke({})
    
    assert result["status"] == "success"
    
    # 1. Text length saved in full_report
    full_report = session.results_store[result["result_id"]]["full_report"]
    assert list(full_report["text_length"]) == [29, 23, 17, 30, 24, 18, 18, 0]
    
    # 2. Overall dataset describe stats
    overall = result["overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0.0, abs=1e-3)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == pytest.approx(30.0, abs=1e-3)
    
    # 3. catA describe stats
    catA_stats = result["by_category"]["catA"]
    assert catA_stats["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA_stats["std"] == pytest.approx(12.5, abs=1e-3)
    
    # 4. catB describe stats
    catB_stats = result["by_category"]["catB"]
    assert catB_stats["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB_stats["std"] == pytest.approx(5.7446, abs=1e-3)
    
    # 5. session.pending_figure is set
    assert session.pending_figure is not None
    
    # 6. JSON serialization works cleanly
    dumped_json = json.dumps(result)
    assert isinstance(dumped_json, str)
