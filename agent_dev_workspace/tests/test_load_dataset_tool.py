import os
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools

def test_load_dataset_tool_fixture():
    # 1. Initialize a single SessionState instance
    session = SessionState()
    
    # 2. Load premade_tools using load_workspace_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    
    # 3. Find load_dataset_tool
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 4. Invoke load_dataset_tool on the fixture dataset
    fixture_path = os.path.join("agent_dev", "sample_fixture.csv")
    result = load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    
    # 5. Assert against known answer #1 in TEST_FIXTURE.md:
    # - Exactly 8 documents
    # - Category counts split {"catA": 4, "catB": 4}
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
