import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session: Any) -> List[Any]:
    @tool
    def cosine_similarity_tool(doc_index_a: int, doc_index_b: int) -> Dict[str, Any]:
        """Compute the cosine similarity between two documents in the session dataset.

        Args:
            doc_index_a: Positional integer index (row offset) of the first document.
            doc_index_b: Positional integer index (row offset) of the second document.

        Returns:
            Dict containing status, result_id, doc_index_a, doc_index_b, and cosine_similarity.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        if "count_vectorizer" not in session.artifacts or session.artifacts["count_vectorizer"] is None:
            return {
                "status": "error",
                "message": "Count vectorizer not found in session artifacts. Please build DTM first."
            }

        vectorizer: Any = session.artifacts["count_vectorizer"]
        n_docs: int = len(session.dataframe)

        if doc_index_a < 0 or doc_index_a >= n_docs or doc_index_b < 0 or doc_index_b >= n_docs:
            return {
                "status": "error",
                "message": f"Document index out of range. Valid range is 0 to {n_docs - 1}."
            }

        text_a: str = str(session.dataframe["text"].iloc[doc_index_a])
        text_b: str = str(session.dataframe["text"].iloc[doc_index_b])

        vec_a: Any = vectorizer.transform([text_a])
        vec_b: Any = vectorizer.transform([text_b])

        sim_matrix: Any = cosine_similarity(vec_a, vec_b)
        sim_val: float = float(sim_matrix[0][0])

        result_id: str = session.next_result_id("cosine")
        summary: Dict[str, Any] = {
            "status": "success",
            "result_id": result_id,
            "doc_index_a": doc_index_a,
            "doc_index_b": doc_index_b,
            "cosine_similarity": sim_val
        }

        session.store_result("cosine_similarity_tool", {"doc_index_a": doc_index_a, "doc_index_b": doc_index_b}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Calculates Pearson feature correlation heatmap and matrix for the top terms by variance.

        Args:
            top_n: Number of top variance terms to evaluate (default 20). Must be > 0.

        Returns:
            Dict containing status, result_id, n_terms, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix has not been built yet. Please run build_dtm_tool first."
            }

        if top_n <= 0:
            return {
                "status": "error",
                "message": "top_n must be a positive integer greater than 0."
            }

        vt: VarianceThreshold = VarianceThreshold()
        vt.fit(session.feature_matrix)
        variances: np.ndarray = vt.variances_

        sorted_indices: np.ndarray = np.argsort(-variances, kind="stable")
        top_indices: np.ndarray = sorted_indices[:top_n]

        selected_terms: List[str] = [str(session.feature_names[i]) for i in top_indices]

        X_top: np.ndarray = session.feature_matrix[:, top_indices].toarray()

        corr_matrix: np.ndarray = np.corrcoef(X_top, rowvar=False)

        if X_top.shape[1] == 1:
            corr_matrix = np.array([[1.0]])
        else:
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)

        fig, ax = plt.subplots(figsize=(max(6, len(selected_terms) * 0.5), max(5, len(selected_terms) * 0.4)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            cmap="coolwarm",
            ax=ax,
            vmin=-1.0,
            vmax=1.0
        )
        ax.set_title(f"Feature Correlation Matrix (Top {len(selected_terms)} Terms by Variance)")
        plt.tight_layout()

        plt.close(fig)
        session.pending_figure = fig

        matrix_list: List[List[float]] = [[float(v) for v in row] for row in corr_matrix]

        full_report_df: pd.DataFrame = pd.DataFrame(
            corr_matrix,
            index=selected_terms,
            columns=selected_terms
        )

        result_id: str = session.next_result_id("feature_correlation")
        summary: Dict[str, Any] = {
            "status": "success",
            "result_id": result_id,
            "n_terms": int(len(selected_terms)),
            "terms": selected_terms,
            "correlation_matrix": matrix_list
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_n": top_n},
            summary,
            full_report=full_report_df
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
