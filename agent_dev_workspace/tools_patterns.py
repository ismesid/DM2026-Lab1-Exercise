import os
import tempfile
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function to create pattern mining tools bound to a session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mines frequent itemsets/patterns for a specific category's documents.

        Args:
            category_name: The category label to filter session.dataframe by.
            filtering_method: Metric for vocabulary filtering ("variance", "tfidf", "term_frequency").
            algorithm: Pattern mining algorithm ("fpgrowth", "topk", "maxfpgrowth").
            min_sup: Minimum support threshold (required integer for fpgrowth/maxfpgrowth).
            k: Top k patterns to retrieve (required integer for topk).

        Returns:
            Dict containing execution status, mined top patterns summary, and result metadata.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please run load_dataset_tool first.",
            }

        if "category_name" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "Column 'category_name' not found in dataframe.",
            }

        cat_df: pd.DataFrame = session.dataframe[
            session.dataframe["category_name"] == category_name
        ]
        if len(cat_df) == 0:
            return {
                "status": "error",
                "message": f"Category '{category_name}' not found in dataframe.",
            }

        valid_algorithms: List[str] = ["fpgrowth", "topk", "maxfpgrowth"]
        if algorithm not in valid_algorithms:
            return {
                "status": "error",
                "message": f"Invalid algorithm '{algorithm}'. Must be one of {valid_algorithms}.",
            }

        valid_filters: List[str] = ["variance", "tfidf", "term_frequency"]
        if filtering_method not in valid_filters:
            return {
                "status": "error",
                "message": f"Invalid filtering_method '{filtering_method}'. Must be one of {valid_filters}.",
            }

        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or type(min_sup) is not int or min_sup <= 0:
                return {
                    "status": "error",
                    "message": f"Parameter 'min_sup' must be a positive integer for algorithm '{algorithm}'.",
                }

        if algorithm == "topk":
            if k is None or type(k) is not int or k <= 0:
                return {
                    "status": "error",
                    "message": "Parameter 'k' must be a positive integer for algorithm 'topk'.",
                }

        result_id: str = session.next_result_id("mine_patterns")
        args: Dict[str, Any] = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }

        texts: List[str] = cat_df["text"].fillna("").astype(str).tolist()
        vectorizer: CountVectorizer = CountVectorizer(stop_words="english")

        try:
            dtm_sparse: Any = vectorizer.fit_transform(texts)
        except ValueError:
            dtm_sparse = None

        if dtm_sparse is None or dtm_sparse.shape[1] == 0:
            summary: Dict[str, Any] = {
                "status": "success",
                "result_id": result_id,
                "category": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns_count": 0,
                "top_patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", args, summary)
            return summary

        dtm_dense: np.ndarray = dtm_sparse.toarray()
        feature_names: List[str] = list(vectorizer.get_feature_names_out())

        kept_indices: List[int] = []
        if filtering_method == "variance":
            variances: np.ndarray = np.var(dtm_dense, axis=0)
            p5: float = float(np.percentile(variances, 5))
            p95: float = float(np.percentile(variances, 95))
            kept_indices = [
                idx
                for idx, v in enumerate(variances)
                if p5 <= float(v) <= p95
            ]
        elif filtering_method == "term_frequency":
            counts: np.ndarray = np.sum(dtm_dense, axis=0)
            p5 = float(np.percentile(counts, 5))
            p95 = float(np.percentile(counts, 95))
            kept_indices = [
                idx
                for idx, c in enumerate(counts)
                if p5 <= float(c) <= p95
            ]
        elif filtering_method == "tfidf":
            transformer: TfidfTransformer = TfidfTransformer()
            tfidf_dense: np.ndarray = transformer.fit_transform(
                dtm_dense
            ).toarray()
            mean_tfidf: np.ndarray = np.mean(tfidf_dense, axis=0)
            p20: float = float(np.percentile(mean_tfidf, 20))
            kept_indices = [
                idx
                for idx, t in enumerate(mean_tfidf)
                if float(t) >= p20
            ]

        if len(kept_indices) == 0:
            summary = {
                "status": "success",
                "result_id": result_id,
                "category": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns_count": 0,
                "top_patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", args, summary)
            return summary

        kept_terms: List[str] = [feature_names[i] for i in kept_indices]
        filtered_dtm: np.ndarray = dtm_dense[:, kept_indices]

        filtered_df: pd.DataFrame = pd.DataFrame(
            filtered_dtm, columns=kept_terms
        )

        mined_patterns: List[Dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            db_txt: str = os.path.join(temp_dir, "db.txt")

            from PAMI.extras.convert import DF2DB

            converter: DF2DB.DF2DB = DF2DB.DF2DB(
                filtered_df, DFtype="dense"
            )
            converter.convert2TransactionalDatabase(db_txt, ">=", 1)

            miner: Any = None
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth

                miner = FPGrowth.FPGrowth(db_txt, minSup=min_sup)
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE

                miner = FAE.FAE(db_txt, k=k)
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth

                miner = MaxFPGrowth.MaxFPGrowth(db_txt, minSup=min_sup)

            miner.mine()
            patterns_df: pd.DataFrame = miner.getPatternsAsDataFrame()

            if patterns_df is not None and len(patterns_df) > 0:
                for _, row in patterns_df.iterrows():
                    pattern_str: str = str(row["Patterns"])
                    sup_val: int = int(row["Support"])
                    mined_patterns.append(
                        {"pattern": pattern_str, "support": sup_val}
                    )

        mined_patterns.sort(key=lambda x: x["support"], reverse=True)
        artifact_key: str = (
            f"patterns_{category_name}_{filtering_method}_{algorithm}"
        )
        session.artifacts[artifact_key] = mined_patterns

        top_10_patterns: List[Dict[str, Any]] = mined_patterns[:10]
        full_report_df: pd.DataFrame = pd.DataFrame(mined_patterns)

        summary = {
            "status": "success",
            "result_id": result_id,
            "category": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns_count": len(mined_patterns),
            "top_patterns": top_10_patterns,
        }

        session.store_result(
            "mine_patterns_tool", args, summary, full_report=full_report_df
        )
        return summary

    return [mine_patterns_tool]
