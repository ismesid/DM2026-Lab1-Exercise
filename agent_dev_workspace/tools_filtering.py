import pandas as pd
import numpy as np
from typing import List, Dict, Any
from scipy.stats import rankdata
from sklearn.feature_selection import VarianceThreshold
from langchain_core.tools import tool

def make_tools(session: Any) -> List[Any]:

    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Flags near-constant terms by variance for removal without modifying session.feature_matrix.

        Args:
            threshold: Terms with variance greater than this threshold are kept; others are flagged as removed. Must be >= 0.0.

        Returns:
            Dict containing status, result_id, threshold, counts of kept/removed terms, and the first 10 of each list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix has not been built yet. Please run build_dtm_tool first."
            }

        if threshold < 0.0:
            return {
                "status": "error",
                "message": "threshold must be non-negative (>= 0.0)."
            }

        vt: VarianceThreshold = VarianceThreshold()
        vt.fit(session.feature_matrix)
        variances: np.ndarray = vt.variances_

        feature_names: List[str] = session.feature_names

        df_report: pd.DataFrame = pd.DataFrame({
            "term": feature_names,
            "variance": [float(v) for v in variances]
        }).sort_values(by=["variance", "term"], ascending=[False, True]).reset_index(drop=True)

        kept_terms: List[str] = [
            str(term) for term in df_report[df_report["variance"] > threshold]["term"].tolist()
        ]
        removed_terms: List[str] = [
            str(term) for term in df_report[df_report["variance"] <= threshold]["term"].tolist()
        ]

        result_id: str = session.next_result_id("variance_filter")
        summary: Dict[str, Any] = {
            "status": "success",
            "result_id": result_id,
            "threshold": float(threshold),
            "n_terms_kept": int(len(kept_terms)),
            "n_terms_removed": int(len(removed_terms)),
            "kept_terms": kept_terms[:10],
            "removed_terms": removed_terms[:10]
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str
    ) -> Dict[str, Any]:
        """
        Calculates Pearson correlation between every term's frequency and a binary target class (one-vs-rest).

        Args:
            target_class: The target category name to correlate against (one-vs-rest).

        Returns:
            Dict containing status, result_id, target_class, total_terms, and top_10_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix has not been built yet. Please run build_dtm_tool first."
            }

        if session.labels is None:
            return {
                "status": "error",
                "message": "Labels have not been set yet."
            }

        if target_class not in session.categories:
            return {
                "status": "error",
                "message": f"target_class '{target_class}' not found in session categories."
            }

        y: np.ndarray = (np.asarray(session.labels) == target_class).astype(float)

        X: Any = session.feature_matrix
        n_docs: int = int(X.shape[0])

        y_mean: float = float(np.mean(y))
        y_std: float = float(np.std(y))

        col_sums: np.ndarray = np.asarray(X.sum(axis=0)).flatten()
        col_means: np.ndarray = col_sums / n_docs

        col_sq_sums: np.ndarray = np.asarray(X.power(2).sum(axis=0)).flatten()
        col_vars: np.ndarray = (col_sq_sums / n_docs) - (col_means ** 2)
        col_vars = np.maximum(col_vars, 0.0)
        col_stds: np.ndarray = np.sqrt(col_vars)

        xy_prod: np.ndarray = np.asarray(X.T.dot(y)).flatten()
        covariances: np.ndarray = (xy_prod / n_docs) - (col_means * y_mean)

        denom: np.ndarray = col_stds * y_std
        pearson_r: np.ndarray = np.zeros(X.shape[1], dtype=float)
        valid_mask: np.ndarray = denom > 0
        pearson_r[valid_mask] = covariances[valid_mask] / denom[valid_mask]
        pearson_r = np.nan_to_num(pearson_r, nan=0.0)

        df_report: pd.DataFrame = pd.DataFrame({
            "term": [str(term) for term in session.feature_names],
            "pearson_r": [float(r) for r in pearson_r]
        })
        df_report["abs_r"] = df_report["pearson_r"].abs()
        df_report = df_report.sort_values(by=["abs_r", "term"], ascending=[False, True]).drop(columns=["abs_r"]).reset_index(drop=True)

        result_id: str = session.next_result_id("pearson_filter")
        summary: Dict[str, Any] = {
            "status": "success",
            "result_id": result_id,
            "target_class": target_class,
            "total_terms": int(len(df_report)),
            "top_10_terms": df_report.head(10).to_dict(orient="records")
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        max_terms: int = 1500
    ) -> Dict[str, Any]:
        """
        Calculates Spearman rank correlation between term frequencies and a binary target class (one-vs-rest).

        Args:
            target_class: The target category name to correlate against (one-vs-rest).
            max_terms: Maximum number of highest-variance terms to rank and evaluate (default 1500). Must be > 0.

        Returns:
            Dict containing status, result_id, target_class, max_terms, total_terms, and top_10_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix has not been built yet. Please run build_dtm_tool first."
            }

        if session.labels is None:
            return {
                "status": "error",
                "message": "Labels have not been set yet."
            }

        if target_class not in session.categories:
            return {
                "status": "error",
                "message": f"target_class '{target_class}' not found in session categories."
            }

        if max_terms <= 0:
            return {
                "status": "error",
                "message": "max_terms must be positive (> 0)."
            }

        vt: VarianceThreshold = VarianceThreshold()
        vt.fit(session.feature_matrix)
        variances: np.ndarray = vt.variances_

        total_features: int = int(len(variances))
        if total_features <= max_terms:
            selected_indices: np.ndarray = np.arange(total_features)
        else:
            selected_indices = np.argsort(variances)[::-1][:max_terms]

        selected_feature_names: List[str] = [str(session.feature_names[i]) for i in selected_indices]

        X_sub: np.ndarray = session.feature_matrix[:, selected_indices].toarray()

        X_ranked: np.ndarray = rankdata(X_sub, method="average", axis=0)

        y: np.ndarray = (np.asarray(session.labels) == target_class).astype(float)
        n_docs: int = int(X_ranked.shape[0])

        y_mean: float = float(np.mean(y))
        y_std: float = float(np.std(y))

        col_means: np.ndarray = np.mean(X_ranked, axis=0)
        col_stds: np.ndarray = np.std(X_ranked, axis=0)

        covariances: np.ndarray = np.mean(X_ranked * y[:, None], axis=0) - (col_means * y_mean)

        denom: np.ndarray = col_stds * y_std
        spearman_r: np.ndarray = np.zeros(X_ranked.shape[1], dtype=float)
        valid_mask: np.ndarray = denom > 0
        spearman_r[valid_mask] = covariances[valid_mask] / denom[valid_mask]
        spearman_r = np.nan_to_num(spearman_r, nan=0.0)

        df_report: pd.DataFrame = pd.DataFrame({
            "term": selected_feature_names,
            "spearman_r": [float(r) for r in spearman_r]
        })
        df_report["abs_r"] = df_report["spearman_r"].abs()
        df_report = df_report.sort_values(by=["abs_r", "term"], ascending=[False, True]).drop(columns=["abs_r"]).reset_index(drop=True)

        result_id: str = session.next_result_id("spearman_filter")
        summary: Dict[str, Any] = {
            "status": "success",
            "result_id": result_id,
            "target_class": target_class,
            "max_terms": int(max_terms),
            "total_terms": int(len(df_report)),
            "top_10_terms": df_report.head(10).to_dict(orient="records")
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "max_terms": max_terms},
            summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
