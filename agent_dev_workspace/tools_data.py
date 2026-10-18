from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
import pandas as pd
from langchain_core.tools import tool
import nltk

def make_tools(session: Any) -> List[Any]:
    """Factory function creating data tools bound to the given pipeline session."""
    
    @tool
    def check_missing_tool(text_column: str = "text") -> Dict[str, Any]:
        """Checks for missing, null, or empty text values in the working dataset.
        
        Args:
            text_column: Name of the text column to inspect for missing/empty values (default: 'text').
            
        Returns:
            A dictionary summary containing total_documents, missing_count, missing_pct,
            sample_missing_indices (up to 10), result_id, and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please run load_dataset_tool first."
            }
            
        df = session.dataframe
        if text_column not in df.columns:
            return {
                "status": "error",
                "message": f"Column '{text_column}' not found in dataset. Available columns: {list(df.columns)}"
            }
            
        # Detect rows where text is NA/null OR empty/whitespace-only string
        series = df[text_column]
        is_na = series.isna()
        is_empty = series.astype(str).str.strip() == ""
        missing_mask = is_na | is_empty
        
        missing_df = df[missing_mask]
        missing_indices: List[int] = [int(idx) for idx in missing_df.index]
        
        total_docs: int = len(df)
        missing_count: int = len(missing_indices)
        missing_pct: float = round((missing_count / total_docs) * 100, 2) if total_docs > 0 else 0.0
        
        # Summary stays small for the LLM context
        summary: Dict[str, Any] = {
            "result_id": session.next_result_id("check_missing"),
            "total_documents": total_docs,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "sample_missing_indices": missing_indices[:10],
            "status": "success"
        }
        
        # full_report stores the full DataFrame of missing rows
        full_report: pd.DataFrame = missing_df.copy()
        
        session.store_result(
            tool_name="check_missing_tool",
            args={"text_column": text_column},
            summary=summary,
            full_report=full_report
        )
        
        return summary

    @tool
    def check_duplicates_tool(subset_columns: Optional[List[str]] = None, drop: bool = False) -> Dict[str, Any]:
        """Checks for (and optionally drops) duplicate documents in the dataset.
        
        Args:
            subset_columns: List of column names to check for duplicates (default: compares text/scalar columns).
            drop: If True, drops ALL copies of duplicated rows (keep=False) and updates session state.
            
        Returns:
            A dictionary summary containing total_documents, duplicate_count, duplicate_pct,
            sample_duplicate_indices, dropped status, remaining_documents, result_id, and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please run load_dataset_tool first."
            }
            
        df = session.dataframe
        
        # Determine subset columns to evaluate, avoiding unhashable list columns like 'unigrams'
        if subset_columns is not None:
            check_cols: List[str] = subset_columns
        else:
            check_cols = []
            for col in df.columns:
                if col == "unigrams":
                    continue
                # Check sample element type if dataframe is non-empty
                if len(df) > 0 and isinstance(df[col].iloc[0], (list, dict)):
                    continue
                check_cols.append(col)
                
        if not check_cols:
            check_cols = [c for c in df.columns if c != "unigrams"]
            
        # 1. Identify duplicates using keep='first' to accurately count extra duplicate copies
        dup_mask = df.duplicated(subset=check_cols, keep='first')
        dup_indices: List[int] = [int(idx) for idx in df[dup_mask].index]
        
        total_docs: int = len(df)
        duplicate_count: int = len(dup_indices)
        duplicate_pct: float = round((duplicate_count / total_docs) * 100, 2) if total_docs > 0 else 0.0
        
        # Create full report dataframe of duplicated rows
        full_report: pd.DataFrame = df[dup_mask].copy()
        
        remaining_docs: int = total_docs
        
        # 2. If drop=True, drop ALL copies of duplicated rows using keep=False
        if drop:
            cleaned_df: pd.DataFrame = df.drop_duplicates(subset=check_cols, keep=False).reset_index(drop=True)
            session.dataframe = cleaned_df
            remaining_docs = len(cleaned_df)
            
        summary: Dict[str, Any] = {
            "result_id": session.next_result_id("check_duplicates"),
            "total_documents": total_docs,
            "duplicate_count": duplicate_count,
            "duplicate_pct": duplicate_pct,
            "sample_duplicate_indices": dup_indices[:10],
            "dropped": drop,
            "remaining_documents": remaining_docs,
            "status": "success"
        }
        
        session.store_result(
            tool_name="check_duplicates_tool",
            args={"subset_columns": subset_columns, "drop": drop},
            summary=summary,
            full_report=full_report
        )
        
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and folders in the repository or a relative subdirectory.
        
        Args:
            subdirectory: Relative directory path inside the repo to list (default: ".").
            
        Returns:
            A summary dictionary containing result_id, subdirectory, directories, files, and status.
        """
        repo_root: Path = Path.cwd().resolve()
        
        # Resolve path safely relative to repo root
        try:
            target_path: Path = (repo_root / subdirectory).resolve()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Invalid subdirectory path: {str(e)}"
            }
            
        # Check boundary security (must remain inside repo root)
        if not target_path.is_relative_to(repo_root):
            return {
                "status": "error",
                "message": "Access denied: Path is outside the repository root."
            }
            
        # Compute path relative to repo_root
        rel_path: Path = target_path.relative_to(repo_root)
        
        # Check if any relative path component starts with "." (hidden directory/file access)
        for part in rel_path.parts:
            if part.startswith("."):
                return {
                    "status": "error",
                    "message": "Access denied: Accessing hidden directories or files is restricted."
                }
                
        if not target_path.exists():
            return {
                "status": "error",
                "message": f"Directory '{subdirectory}' does not exist."
            }
            
        if not target_path.is_dir():
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' is not a directory."
            }
            
        directories: List[str] = []
        files: List[str] = []
        
        for item in target_path.iterdir():
            # Skip hidden files/directories and __pycache__
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            if item.is_dir():
                directories.append(item.name)
            else:
                files.append(item.name)
                
        summary: Dict[str, Any] = {
            "result_id": session.next_result_id("list_files"),
            "subdirectory": subdirectory,
            "directories": sorted(directories),
            "files": sorted(files),
            "status": "success"
        }
        
        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )
        
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """Subsamples the working DataFrame and replaces session.dataframe with the sample.
        
        Args:
            n: Number of items to sample. Cannot be used together with frac.
            frac: Fraction of items to sample (e.g., 0.5 for 50%). Cannot be used together with n.
            random_state: Random seed for reproducible sampling.
            
        Returns:
            A dictionary summary containing result_id, original_count, sampled_count,
            sample_indices (up to 10 original row indices), sample_indices_count, note (if applicable), and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please run load_dataset_tool first."
            }
            
        df: pd.DataFrame = session.dataframe
        total_docs: int = len(df)
        
        if n is not None and frac is not None:
            return {
                "status": "error",
                "message": "Please specify either 'n' or 'frac', but not both."
            }
            
        if n is None and frac is None:
            return {
                "status": "error",
                "message": "Please specify either 'n' (number of rows) or 'frac' (fraction of rows) to sample."
            }
            
        note: Optional[str] = None
        actual_n: Optional[int] = n
        
        if n is not None:
            if n <= 0:
                return {
                    "status": "error",
                    "message": f"Sample size 'n' must be a positive integer, got {n}."
                }
            if n > total_docs:
                actual_n = total_docs
                note = f"Requested sample size n={n} exceeded total documents ({total_docs}). Capped sample size to {total_docs}."
                
        if frac is not None:
            if frac <= 0.0 or frac > 1.0:
                return {
                    "status": "error",
                    "message": f"Sample fraction 'frac' must be between 0.0 (exclusive) and 1.0 (inclusive), got {frac}."
                }
                
        # Perform sampling
        if actual_n is not None:
            sampled_df: pd.DataFrame = df.sample(n=actual_n, random_state=random_state)
        else:
            sampled_df = df.sample(frac=frac, random_state=random_state)
            
        # Capture original row indices BEFORE resetting index
        original_indices: List[int] = [int(idx) for idx in sampled_df.index]
        
        # full_report preserves the original indices on the sampled DataFrame
        full_report: pd.DataFrame = sampled_df.copy()
        
        # Reset index and replace session.dataframe with the working sample
        sampled_df_reset: pd.DataFrame = sampled_df.reset_index(drop=True)
        session.dataframe = sampled_df_reset
        
        sampled_count: int = len(sampled_df_reset)
        
        summary: Dict[str, Any] = {
            "result_id": session.next_result_id("sample_data"),
            "original_count": total_docs,
            "sampled_count": sampled_count,
            "sample_indices": original_indices[:10],
            "sample_indices_count": len(original_indices),
            "status": "success"
        }
        
        if note:
            summary["note"] = note
            
        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "frac": frac, "random_state": random_state},
            summary=summary,
            full_report=full_report
        )
        
        return summary

    @tool
    def tokenize_tool(text_column: str = "text", lowercase: bool = True) -> Dict[str, Any]:
        """Tokenizes each document's text into unigrams using NLTK word_tokenize.
        
        Args:
            text_column: Column name containing document text to tokenize (default: 'text').
            lowercase: Whether to convert text to lowercase before tokenization (default: True).
            
        Returns:
            A summary dictionary containing total_documents, total_tokens, vocabulary_size,
            avg_tokens_per_doc, sample_tokens, result_id, and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please run load_dataset_tool first."
            }
            
        df: pd.DataFrame = session.dataframe
        if text_column not in df.columns:
            return {
                "status": "error",
                "message": f"Column '{text_column}' not found in dataset. Available columns: {list(df.columns)}"
            }
            
        # Ensure required NLTK tokenizers are downloaded
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            
        # Tokenize each document
        unigrams_list: List[List[str]] = []
        all_tokens: List[str] = []
        
        for text_val in df[text_column]:
            if pd.isna(text_val):
                text_str: str = ""
            else:
                text_str = str(text_val)
                
            if lowercase:
                text_str = text_str.lower()
                
            tokens: List[str] = nltk.word_tokenize(text_str) if text_str.strip() else []
            unigrams_list.append(tokens)
            all_tokens.extend(tokens)
            
        # Save unigrams list column to session dataframe
        session.dataframe["unigrams"] = unigrams_list
        
        total_docs: int = len(df)
        total_tokens: int = len(all_tokens)
        vocab_size: int = len(set(all_tokens))
        avg_tokens: float = round(total_tokens / total_docs, 2) if total_docs > 0 else 0.0
        
        summary: Dict[str, Any] = {
            "result_id": session.next_result_id("tokenize"),
            "total_documents": total_docs,
            "total_tokens": total_tokens,
            "vocabulary_size": vocab_size,
            "avg_tokens_per_doc": avg_tokens,
            "sample_tokens": unigrams_list[0][:10] if unigrams_list and unigrams_list[0] else [],
            "status": "success"
        }
        
        # full_report stores a DataFrame containing the token lists for every document
        full_report: pd.DataFrame = pd.DataFrame({"unigrams": unigrams_list})
        
        session.store_result(
            tool_name="tokenize_tool",
            args={"text_column": text_column, "lowercase": lowercase},
            summary=summary,
            full_report=full_report
        )
        
        return summary

    @tool
    def inspect_data_tool(
        position: Literal["first", "last"] = "first",
        n: int = 5,
        text_column: str = "text",
        category_column: str = "category_name"
    ) -> Dict[str, Any]:
        """Peeks at the first or last n rows of each category in the working dataset.
        
        Args:
            position: Whether to show the 'first' or 'last' rows of each category (default: 'first').
            n: Number of rows to display per category (default: 5).
            text_column: Name of the text column to inspect and check for missing values (default: 'text').
            category_column: Name of the column defining document categories (default: 'category_name').
            
        Returns:
            A summary dictionary containing result_id, total_documents, position, requested_n,
            categories (with per-category missing text counts, row counts, capped notes, and truncated row samples),
            and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please run load_dataset_tool first."
            }
            
        if n <= 0:
            return {
                "status": "error",
                "message": f"Row count 'n' must be a positive integer, got {n}."
            }
            
        df: pd.DataFrame = session.dataframe
        if text_column not in df.columns:
            return {
                "status": "error",
                "message": f"Column '{text_column}' not found in dataset. Available columns: {list(df.columns)}"
            }
            
        total_docs: int = len(df)
        
        # Determine how to group categories
        if category_column in df.columns:
            categories_groups = df.groupby(category_column, observed=False, sort=False)
        else:
            categories_groups = [("all", df)]
            
        category_summaries: Dict[str, Any] = {}
        inspected_rows_list: List[pd.DataFrame] = []
        
        for cat_name, cat_df in categories_groups:
            cat_name_str: str = str(cat_name)
            cat_total: int = len(cat_df)
            
            # Count missing/empty texts in this category
            series = cat_df[text_column]
            is_na = series.isna()
            is_empty = series.astype(str).str.strip() == ""
            missing_count: int = int((is_na | is_empty).sum())
            
            note: Optional[str] = None
            if n > cat_total:
                actual_n: int = cat_total
                note = f"Requested n={n} exceeds category size ({cat_total}). Displaying all {cat_total} rows."
            else:
                actual_n = n
                
            slice_df: pd.DataFrame = cat_df.head(actual_n) if position == "first" else cat_df.tail(actual_n)
            inspected_rows_list.append(slice_df)
            
            # Format rows for summary: exclude unigrams, truncate text to first 100 chars
            summary_samples: List[Dict[str, Any]] = []
            for idx, row in slice_df.iterrows():
                text_val = row[text_column]
                if pd.isna(text_val):
                    text_str: str = "<MISSING>"
                else:
                    raw_str: str = str(text_val)
                    text_str = raw_str[:100] + ("..." if len(raw_str) > 100 else "")
                    
                sample_dict: Dict[str, Any] = {
                    "index": int(idx),
                    text_column: text_str
                }
                
                # Include other scalar metadata columns, converting scalar types to native Python types
                for col in slice_df.columns:
                    if col in (text_column, "unigrams"):
                        continue
                    val = row[col]
                    if pd.isna(val):
                        sample_dict[col] = None
                    elif hasattr(val, "item"):
                        sample_dict[col] = val.item()
                    elif not isinstance(val, (list, dict)):
                        sample_dict[col] = val
                        
                summary_samples.append(sample_dict)
                
            cat_summary: Dict[str, Any] = {
                "total_documents": cat_total,
                "missing_text_count": missing_count,
                "sample_rows": summary_samples
            }
            if note:
                cat_summary["note"] = note
                
            category_summaries[cat_name_str] = cat_summary
            
        full_report: pd.DataFrame = pd.concat(inspected_rows_list) if inspected_rows_list else pd.DataFrame()
        
        summary: Dict[str, Any] = {
            "result_id": session.next_result_id("inspect_data"),
            "total_documents": total_docs,
            "position": position,
            "requested_n": n,
            "categories": category_summaries,
            "status": "success"
        }
        
        session.store_result(
            tool_name="inspect_data_tool",
            args={
                "position": position,
                "n": n,
                "text_column": text_column,
                "category_column": category_column
            },
            summary=summary,
            full_report=full_report
        )
        
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes summary statistics on document text length and generates a box plot grouped by category.
        
        Returns:
            A dictionary summary containing result_id, overall statistics, by_category statistics, and status.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please run load_dataset_tool first."
            }

        df: pd.DataFrame = session.dataframe.copy()
        if "text" not in df.columns:
            return {
                "status": "error",
                "message": "Column 'text' not found in dataset."
            }

        # Compute character length of document text
        df["text_length"] = df["text"].fillna("").astype(str).apply(len)

        def clean_describe_dict(series: pd.Series) -> Dict[str, Any]:
            """Converts .describe() series to native Python types, replacing NaN/Inf with None."""
            desc: pd.Series = series.describe()
            cleaned: Dict[str, Any] = {}
            for stat, val in desc.items():
                stat_str: str = str(stat)
                if pd.isna(val) or np.isinf(val):
                    cleaned[stat_str] = None
                elif stat_str == "count":
                    cleaned[stat_str] = int(val)
                else:
                    cleaned[stat_str] = float(val)
            return cleaned

        overall_stats: Dict[str, Any] = clean_describe_dict(df["text_length"])

        # Determine category column to group by
        category_col: Optional[str] = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        by_category_stats: Dict[str, Any] = {}
        if category_col:
            for cat_val, grp in df.groupby(category_col, observed=False):
                cat_str: str = str(cat_val)
                by_category_stats[cat_str] = clean_describe_dict(grp["text_length"])

        # Generate box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        if category_col:
            sns.boxplot(data=df, x=category_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
        
        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Document Text Length Distribution by Category")
        fig.tight_layout()

        # Close figure before binding to pending_figure to prevent duplicate notebook rendering
        plt.close(fig)
        session.pending_figure = fig

        summary: Dict[str, Any] = {
            "result_id": session.next_result_id("describe_data"),
            "overall": overall_stats,
            "by_category": by_category_stats,
            "status": "success"
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary,
            full_report=df[["text", "text_length"] + ([category_col] if category_col else [])]
        )

        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        list_files_tool,
        sample_data_tool,
        tokenize_tool,
        inspect_data_tool,
        describe_data_tool
    ]
