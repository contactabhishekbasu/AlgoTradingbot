"""Database connection and operations for PostgreSQL."""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool, QueuePool

from utils.config import settings
from utils.logger import logger


class DatabaseClient:
    """
    PostgreSQL database client with connection pooling.

    Features:
    - Connection pooling
    - Transaction management
    - Pandas integration
    - Error handling
    - Query logging
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
    ):
        """
        Initialize database client.

        Args:
            database_url: Database connection URL (default from settings)
            pool_size: Connection pool size
            max_overflow: Maximum pool overflow
            pool_timeout: Pool timeout in seconds
        """
        self.database_url = database_url or settings.database_url

        # Create engine with connection pooling
        self.engine: Engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=True,  # Verify connections before using
            echo=settings.debug,  # Log SQL in debug mode
        )

        logger.info(
            "database_client_initialized",
            database=self._mask_url(self.database_url),
            pool_size=pool_size,
            max_overflow=max_overflow,
        )

    def _mask_url(self, url: str) -> str:
        """Mask password in database URL for logging."""
        if "@" in url:
            parts = url.split("@")
            credentials = parts[0].split("://")[1]
            if ":" in credentials:
                user = credentials.split(":")[0]
                return url.replace(credentials, f"{user}:***")
        return url

    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool.

        Yields:
            SQLAlchemy connection

        Example:
            with db.get_connection() as conn:
                result = conn.execute("SELECT * FROM table")
        """
        connection = self.engine.connect()
        try:
            yield connection
            connection.commit()
        except Exception as e:
            connection.rollback()
            logger.error("database_transaction_failed", error=str(e))
            raise
        finally:
            connection.close()

    def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """
        Execute a SQL query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query result

        Raises:
            SQLAlchemyError: If query execution fails
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query), params or {})
                logger.debug("query_executed", query=query[:100])
                return result

        except SQLAlchemyError as e:
            logger.error(
                "query_execution_failed",
                query=query[:100],
                error=str(e),
            )
            raise

    def fetch_df(
        self,
        query: str,
        params: Optional[Dict] = None,
        index_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Execute query and return result as DataFrame.

        Args:
            query: SQL query string
            params: Query parameters
            index_col: Column to use as index

        Returns:
            DataFrame with query results
        """
        try:
            df = pd.read_sql(
                query,
                self.engine,
                params=params,
                index_col=index_col,
            )

            logger.debug(
                "dataframe_fetched",
                query=query[:100],
                rows=len(df),
                columns=len(df.columns),
            )

            return df

        except SQLAlchemyError as e:
            logger.error(
                "dataframe_fetch_failed",
                query=query[:100],
                error=str(e),
            )
            raise

    def insert_df(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = "append",
        index: bool = False,
    ) -> int:
        """
        Insert DataFrame into database table.

        Args:
            df: DataFrame to insert
            table: Table name
            if_exists: Action if table exists ('fail', 'replace', 'append')
            index: Write DataFrame index as column

        Returns:
            Number of rows inserted
        """
        try:
            rows = df.to_sql(
                table,
                self.engine,
                if_exists=if_exists,
                index=index,
                method="multi",  # Batch insert for better performance
            )

            logger.info(
                "dataframe_inserted",
                table=table,
                rows=rows or len(df),
                if_exists=if_exists,
            )

            return rows or len(df)

        except SQLAlchemyError as e:
            logger.error(
                "dataframe_insert_failed",
                table=table,
                rows=len(df),
                error=str(e),
            )
            raise

    def store_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
    ) -> int:
        """
        Store market data in database.

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol

        Returns:
            Number of rows inserted
        """
        # Prepare data
        df = data.copy()

        # Ensure required columns
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Rename columns to match database schema
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Add symbol if not present
        if "symbol" not in df.columns:
            df["symbol"] = symbol

        # Select only the columns we need
        df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

        # Insert data (ignore duplicates)
        query = """
        INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
        VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            updated_at = NOW()
        """

        inserted = 0
        with self.get_connection() as conn:
            for _, row in df.iterrows():
                try:
                    conn.execute(text(query), row.to_dict())
                    inserted += 1
                except SQLAlchemyError as e:
                    logger.warning(
                        "market_data_insert_failed",
                        symbol=symbol,
                        timestamp=row["timestamp"],
                        error=str(e),
                    )

        logger.info(
            "market_data_stored",
            symbol=symbol,
            rows_inserted=inserted,
            rows_total=len(df),
        )

        return inserted

    def fetch_market_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch market data from database.

        Args:
            symbol: Stock symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            limit: Maximum number of rows

        Returns:
            DataFrame with OHLCV data
        """
        query = """
        SELECT
            timestamp as Date,
            open as Open,
            high as High,
            low as Low,
            close as Close,
            volume as Volume
        FROM market_data
        WHERE symbol = :symbol
        """

        params = {"symbol": symbol}

        if start:
            query += " AND timestamp >= :start"
            params["start"] = start

        if end:
            query += " AND timestamp <= :end"
            params["end"] = end

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        df = self.fetch_df(query, params)

        logger.info(
            "market_data_fetched",
            symbol=symbol,
            rows=len(df),
            start=start,
            end=end,
        )

        return df

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("database_connection_test_passed")
            return True

        except SQLAlchemyError as e:
            logger.error("database_connection_test_failed", error=str(e))
            return False

    def get_table_info(self, table: str) -> Dict[str, Any]:
        """
        Get information about a table.

        Args:
            table: Table name

        Returns:
            Dictionary with table information
        """
        try:
            # Validate table name to prevent SQL injection
            # Only allow alphanumeric characters and underscores
            if not table.replace('_', '').isalnum():
                raise ValueError(f"Invalid table name: {table}")

            # Get row count
            # Note: Table names cannot be parameterized, but we've validated the input
            count_query = f"SELECT COUNT(*) as count FROM {table}"  # nosec B608
            result = self.execute(count_query)
            row_count = result.fetchone()[0]

            # Get column information
            columns_query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = :table
            ORDER BY ordinal_position
            """
            columns_df = self.fetch_df(columns_query, {"table": table})

            info = {
                "table": table,
                "row_count": row_count,
                "columns": columns_df.to_dict("records"),
            }

            logger.debug("table_info_fetched", table=table, row_count=row_count)
            return info

        except SQLAlchemyError as e:
            logger.error("table_info_fetch_failed", table=table, error=str(e))
            raise

    def close(self):
        """Close database connection pool."""
        self.engine.dispose()
        logger.info("database_connection_pool_closed")
