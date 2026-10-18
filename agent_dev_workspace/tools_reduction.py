import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Literal, Optional
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarize category labels into a one-hot encoded matrix.

        Reads session.labels and session.categories, performs one-hot encoding using
        pd.get_dummies, stores the resulting DataFrame in
        session.artifacts["binarized_labels"], and records the result.

        Returns:
            Dict[str, Any]: Summary containing status, result_id, categories, shape,
            category counts, and the first 10 rows.
        """
        if session.labels is None or session.categories is None:
            return {
                "status": "error",
                "message": "session.labels or session.categories is None. Run build_dtm_tool or set labels first."
            }

        result_id: str = session.next_result_id("binarize_labels")

        # Ensure categories follow session.categories order
        cat_series: pd.Categorical = pd.Categorical(session.labels, categories=session.categories)
        df_onehot: pd.DataFrame = pd.get_dummies(cat_series, dtype=int)
        
        # Ensure column names match string categories explicitly
        df_onehot.columns = [str(c) for c in session.categories]

        # Store full DataFrame artifact in session
        session.artifacts["binarized_labels"] = df_onehot

        # Calculate counts per category
        counts_dict: Dict[str, int] = {str(k): int(v) for k, v in df_onehot.sum(axis=0).items()}

        # First 10 rows as serializable dicts
        head_rows: List[Dict[str, Any]] = df_onehot.head(10).to_dict(orient="records")

        summary: Dict[str, Any] = {
            "status": "success",
            "result_id": result_id,
            "categories": [str(c) for c in session.categories],
            "shape": [int(df_onehot.shape[0]), int(df_onehot.shape[1])],
            "counts": counts_dict,
            "head": head_rows
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
            full_report=df_onehot
        )

        return summary

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"],
        n_components: int = 2,
        random_state: int = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """Performs dimensionality reduction (PCA, t-SNE, or UMAP) on session.feature_matrix.

        Generates a 2D scatter plot colored by session.labels, sets it on session.pending_figure,
        and stores the reduced coordinates in session.artifacts[f"reduced_{method}"].

        Args:
            method: Reduction technique to apply ("pca", "tsne", or "umap").
            n_components: Target dimensions for reduction (default 2, must be >= 2 for plotting).
            random_state: Seed for reproducible output (default 42).
            perplexity: Perplexity setting for t-SNE (default 30.0).
            n_neighbors: Number of neighbors setting for UMAP (default 15).

        Returns:
            Dict[str, Any]: Summary containing status, result_id, method, n_components,
            first 10 reduced coordinates as a list of float lists, and explained_variance_ratio (for PCA).
        """
        if session.feature_matrix is None:
            return {
                "status": "error",
                "message": "session.feature_matrix is None. Run build_dtm_tool first."
            }

        if session.labels is None:
            return {
                "status": "error",
                "message": "session.labels is None. Labels are required for visualization."
            }

        if n_components < 2:
            return {
                "status": "error",
                "message": f"n_components must be at least 2 for scatter plotting, got {n_components}."
            }

        n_samples: int = session.feature_matrix.shape[0]

        if method == "tsne" and perplexity >= n_samples:
            return {
                "status": "error",
                "message": f"perplexity ({perplexity}) must be less than number of documents ({n_samples})."
            }

        if method == "umap" and n_neighbors >= n_samples:
            return {
                "status": "error",
                "message": f"n_neighbors ({n_neighbors}) must be less than number of documents ({n_samples})."
            }

        X_dense: np.ndarray = session.feature_matrix.toarray()
        coords: np.ndarray
        explained_variance_ratio: Optional[List[float]] = None

        if method == "pca":
            from sklearn.decomposition import PCA
            pca: PCA = PCA(n_components=n_components, random_state=random_state)
            coords = pca.fit_transform(X_dense)
            explained_variance_ratio = [float(v) for v in pca.explained_variance_ratio_]
        elif method == "tsne":
            from sklearn.manifold import TSNE
            tsne: TSNE = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords = tsne.fit_transform(X_dense)
        elif method == "umap":
            try:
                import umap
            except ImportError:
                return {
                    "status": "error",
                    "message": "umap-learn package is not installed. Please install umap-learn to use UMAP."
                }
            umap_model: Any = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = umap_model.fit_transform(X_dense)
        else:
            return {
                "status": "error",
                "message": f"Unknown method '{method}'. Supported methods are 'pca', 'tsne', and 'umap'."
            }

        # Save coordinates in artifacts with method-specific key
        session.artifacts[f"reduced_{method}"] = coords

        # Generate scatter plot colored by session.labels
        labels_arr: np.ndarray = np.array(session.labels)
        unique_labels: np.ndarray = np.unique(labels_arr)

        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl in unique_labels:
            mask: np.ndarray = (labels_arr == lbl)
            ax.scatter(coords[mask, 0], coords[mask, 1], label=str(lbl), alpha=0.7)

        ax.set_title(f"{method.upper()} Dimensionality Reduction")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()
        plt.close(fig)

        session.pending_figure = fig

        # Prepare summary and full report
        result_id: str = session.next_result_id("reduce_dimensions")
        head_coords: List[List[float]] = [[float(val) for val in row[:2]] for row in coords[:10]]

        df_coords: pd.DataFrame = pd.DataFrame(
            coords,
            columns=[f"Dim{i+1}" for i in range(coords.shape[1])]
        )
        df_coords["label"] = labels_arr

        summary: Dict[str, Any] = {
            "status": "success",
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": head_coords
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors
            },
            summary=summary,
            full_report=df_coords
        )

        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
