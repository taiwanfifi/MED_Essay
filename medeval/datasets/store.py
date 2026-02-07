"""SQLite storage layer for MedEval-X questions."""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from medeval.config import DB_PATH
from medeval.datasets.schema import MedicalQuestion

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    question TEXT NOT NULL,
    options_json TEXT,
    correct_answer TEXT NOT NULL,
    topic TEXT,
    difficulty TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_dataset ON questions(dataset);",
    "CREATE INDEX IF NOT EXISTS idx_topic ON questions(topic);",
    "CREATE INDEX IF NOT EXISTS idx_split ON questions(split);",
    "CREATE INDEX IF NOT EXISTS idx_dataset_split ON questions(dataset, split);",
]

INSERT_SQL = """
INSERT OR REPLACE INTO questions
    (id, dataset, split, question, options_json, correct_answer, topic, difficulty, metadata_json)
VALUES
    (:id, :dataset, :split, :question, :options_json, :correct_answer, :topic, :difficulty, :metadata_json);
"""


class QuestionStore:
    """SQLite-backed store for MedicalQuestion objects."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        for idx_sql in CREATE_INDEXES_SQL:
            cursor.execute(idx_sql)
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_questions(self, questions: List[MedicalQuestion]) -> Tuple[int, Optional[str]]:
        """Insert a batch of questions into the database.

        Returns:
            (count_inserted, error) tuple
        """
        if not questions:
            return 0, None

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            rows = [q.to_dict() for q in questions]
            cursor.executemany(INSERT_SQL, rows)
            conn.commit()
            count = cursor.rowcount
            conn.close()
            logger.info(f"Inserted {len(questions)} questions")
            return len(questions), None
        except Exception as e:
            error = f"Failed to insert questions: {e}"
            logger.error(error)
            return 0, error

    def count_by_dataset(self) -> Dict[str, int]:
        """Get question counts grouped by dataset."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT dataset, COUNT(*) as cnt FROM questions GROUP BY dataset")
        result = {row["dataset"]: row["cnt"] for row in cursor.fetchall()}
        conn.close()
        return result

    def count_by_dataset_split(self) -> Dict[str, Dict[str, int]]:
        """Get question counts grouped by dataset and split."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT dataset, split, COUNT(*) as cnt FROM questions GROUP BY dataset, split"
        )
        result: Dict[str, Dict[str, int]] = {}
        for row in cursor.fetchall():
            ds = row["dataset"]
            if ds not in result:
                result[ds] = {}
            result[ds][row["split"]] = row["cnt"]
        conn.close()
        return result

    def count_by_topic(self, dataset: Optional[str] = None) -> Dict[str, int]:
        """Get question counts grouped by topic."""
        conn = self._get_conn()
        cursor = conn.cursor()
        if dataset:
            cursor.execute(
                "SELECT topic, COUNT(*) as cnt FROM questions WHERE dataset = ? GROUP BY topic ORDER BY cnt DESC",
                (dataset,),
            )
        else:
            cursor.execute(
                "SELECT topic, COUNT(*) as cnt FROM questions GROUP BY topic ORDER BY cnt DESC"
            )
        result = {row["topic"] or "(none)": row["cnt"] for row in cursor.fetchall()}
        conn.close()
        return result

    def get_total_count(self) -> int:
        """Get total number of questions."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM questions")
        count = cursor.fetchone()["cnt"]
        conn.close()
        return count

    def get_questions(
        self,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MedicalQuestion]:
        """Query questions with optional filters."""
        conn = self._get_conn()
        cursor = conn.cursor()

        conditions = []
        params = []
        if dataset:
            conditions.append("dataset = ?")
            params.append(dataset)
        if split:
            conditions.append("split = ?")
            params.append(split)
        if topic:
            conditions.append("topic = ?")
            params.append(topic)

        sql = "SELECT * FROM questions"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        if limit:
            sql += f" LIMIT {limit}"

        cursor.execute(sql, params)
        questions = [MedicalQuestion.from_dict(dict(row)) for row in cursor.fetchall()]
        conn.close()
        return questions

    def clear(self):
        """Delete all questions from the database."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM questions")
        conn.commit()
        conn.close()
        logger.info("All questions cleared from database")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    store = QuestionStore()
    print(f"Total questions: {store.get_total_count()}")
    print(f"By dataset: {store.count_by_dataset()}")
    print(f"By dataset/split: {store.count_by_dataset_split()}")

    # Quick insert test
    from medeval.datasets.schema import MedicalQuestion

    test_q = MedicalQuestion(
        id="test_0",
        dataset="test",
        split="test",
        question="What is the capital of France?",
        options={"A": "London", "B": "Paris", "C": "Berlin", "D": "Madrid"},
        correct_answer="B",
        topic="geography",
    )
    count, err = store.insert_questions([test_q])
    print(f"Inserted: {count}, error: {err}")
    print(f"Total after insert: {store.get_total_count()}")

    retrieved = store.get_questions(dataset="test")
    print(f"Retrieved: {retrieved}")
