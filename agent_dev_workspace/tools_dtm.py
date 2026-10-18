import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Union
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool
from scipy.sparse import csr_matrix

def make_tools(session: Any) -> List[Any]:

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        max_df: float = 1.0
    ) -> Dict[str, Any]:
        """
        Builds a Document-Term Matrix (DTM) from the working dataset text column using CountVectorizer.

        Args:
            ngram_range: A list of two integers [min_n, max_n] specifying the range of n-gram sizes.
            min_df: Minimum document frequency threshold as an absolute document count.
            max_df: Maximum document frequency threshold as a proportion of documents.

        Returns:
            Dict summarizing matrix dimensions, vocabulary size, and sparsity.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded or 'text' column missing."}

        # Validate ngram_range parameter
        if len(ngram_range) != 2:
            return {"status": "error", "message": "ngram_range must be a list of exactly two integers [min_n, max_n]."}
        
        min_n: int = ngram_range[0]
        max_n: int = ngram_range[1]
        if min_n < 1 or max_n < 1:
            return {"status": "error", "message": "ngram_range values must both be >= 1."}
        if min_n > max_n:
            return {"status": "error", "message": "The first value in ngram_range cannot be greater than the second value."}

        try:
            vectorizer: CountVectorizer = CountVectorizer(
                ngram_range=(min_n, max_n),
                min_df=min_df,
                max_df=max_df
            )
            text_series: pd.Series = session.dataframe["text"].fillna("").astype(str)
            feature_matrix: csr_matrix = vectorizer.fit_transform(text_series)
        except ValueError as e:
            return {"status": "error", "message": f"CountVectorizer failed (e.g. empty vocabulary): {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"Error building DTM: {str(e)}"}

        feature_names: List[str] = vectorizer.get_feature_names_out().tolist()

        if len(feature_names) == 0 or feature_matrix.shape[1] == 0:
            return {"status": "error", "message": "CountVectorizer resulted in an empty vocabulary."}

        # Store in session state
        session.artifacts["count_vectorizer"] = vectorizer
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        n_rows: int = feature_matrix.shape[0]
        n_cols: int = feature_matrix.shape[1]
        non_zero: int = int(feature_matrix.nnz)
        total_elements: int = int(n_rows * n_cols)
        sparsity_pct: float = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        result_id: str = session.next_result_id("build_dtm")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range, "min_df": min_df, "max_df": max_df}, summary)
        return summary

    @tool
    def term_frequency_tool(
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Aggregates term frequencies across all documents in the Document-Term Matrix (DTM).

        Args:
            top_n: Number of top frequent terms to return in the summary. Must be > 0.

        Returns:
            Dict containing the status, total vocabulary size, requested top_n, and top_terms mapping.
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

        counts: List[int] = [int(c) for c in np.asarray(session.feature_matrix.sum(axis=0)).ravel()]

        df_report: pd.DataFrame = pd.DataFrame({
            "term": session.feature_names,
            "frequency": counts
        }).sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(drop=True)

        top_df: pd.DataFrame = df_report.head(top_n)
        top_terms_dict: Dict[str, int] = {
            str(row["term"]): int(row["frequency"]) for _, row in top_df.iterrows()
        }

        result_id: str = session.next_result_id("term_frequency")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "status": "success",
            "total_terms": len(session.feature_names),
            "top_n": top_n,
            "top_terms": top_terms_dict
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=df_report)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """
        Generates a heatmap visualization and numerical slice of the raw Document-Term Matrix (DTM).

        Args:
            n_terms: Number of terms (columns) to include from the DTM, in original order. Must be > 0.
            n_documents: Number of documents (rows) to include from the DTM, in original row order. Must be > 0.

        Returns:
            Dict containing status, sliced matrix, term labels, document labels, and result metadata.
        """
        if n_terms <= 0 or n_documents <= 0:
            return {
                "status": "error",
                "message": "n_terms and n_documents must be positive integers greater than 0."
            }

        if session.feature_matrix is None or session.feature_names is None or session.dataframe is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix or dataset has not been loaded/built. Please run build_dtm_tool first."
            }

        total_docs: int = int(session.feature_matrix.shape[0])
        total_terms: int = int(session.feature_matrix.shape[1])

        actual_n_docs: int = min(n_documents, total_docs)
        actual_n_terms: int = min(n_terms, total_terms)

        raw_slice: np.ndarray = session.feature_matrix[0:actual_n_docs, 0:actual_n_terms].toarray()

        doc_labels: List[int] = [int(idx) for idx in session.dataframe.index[:actual_n_docs]]
        term_labels: List[str] = [str(term) for term in session.feature_names[:actual_n_terms]]

        full_report_df: pd.DataFrame = pd.DataFrame(
            raw_slice,
            index=doc_labels,
            columns=term_labels
        )

        fig, ax = plt.subplots(figsize=(max(6, actual_n_terms * 0.5), max(4, actual_n_docs * 0.4)))
        sns.heatmap(
            raw_slice,
            annot=True,
            fmt="d",
            xticklabels=term_labels,
            yticklabels=[str(d) for d in doc_labels],
            cmap="Blues",
            ax=ax
        )
        ax.set_title(f"DTM Heatmap Slice ({actual_n_docs} Docs x {actual_n_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        plt.close(fig)
        session.pending_figure = fig

        matrix_slice_list: List[List[int]] = [[int(val) for val in row] for row in raw_slice]

        result_id: str = session.next_result_id("dtm_heatmap")
        summary: Dict[str, Any] = {
            "status": "success",
            "result_id": result_id,
            "n_documents_sliced": actual_n_docs,
            "n_terms_sliced": actual_n_terms,
            "doc_labels": doc_labels,
            "term_labels": term_labels,
            "matrix_slice": matrix_slice_list
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary, full_report=full_report_df)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
