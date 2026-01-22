"""
Data Loading Module
Handles loading and merging data from Excel sheets with comprehensive error handling.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from ..utils.logger import get_logger
from ..utils.config import get_config


class DataLoader:
    """Load and merge school shooting incident data from Excel file."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataLoader.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config()
        self.logger = get_logger(__name__)

    def load_excel_file(self, file_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all required sheets from Excel file.

        Args:
            file_path: Path to Excel file. If None, uses config default

        Returns:
            Dictionary of sheet_name: DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required sheets are missing
        """
        if file_path is None:
            file_path = self.config.get("data.input_excel")

        file_path = Path(file_path)

        # Validate file existence
        if not file_path.exists():
            error_msg = f"Data file not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info(f"Loading data from: {file_path}")

        try:
            # Load Excel file
            xls = pd.ExcelFile(file_path)

            # Get required sheets from config
            required_sheets = self.config.get("data.required_sheets", [])
            sheet_names = self.config.get("data.sheets", {})

            # Validate sheets exist
            missing_sheets = set(required_sheets) - set(xls.sheet_names)
            if missing_sheets:
                error_msg = f"Missing required sheets: {missing_sheets}. Found: {xls.sheet_names}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Load each sheet
            data = {}
            for sheet_key, sheet_name in sheet_names.items():
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    data[sheet_key] = df
                    self.logger.info(
                        f"Loaded sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns"
                    )
                except Exception as e:
                    error_msg = f"Error loading sheet '{sheet_name}': {str(e)}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            self.logger.info(f"Successfully loaded {len(data)} sheets")
            return data

        except Exception as e:
            error_msg = f"Failed to load Excel file: {str(e)}"
            self.logger.error(error_msg)
            raise

    def aggregate_categorical(self, series: pd.Series) -> str:
        """
        Aggregate categorical values by joining unique values.

        Args:
            series: Pandas series with categorical data

        Returns:
            Semicolon-separated string of unique values
        """
        return ";".join(series.dropna().astype(str).unique())

    def aggregate_shooter_data(self, shooter_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate shooter data by Incident_ID.

        Args:
            shooter_df: Shooter DataFrame

        Returns:
            Aggregated shooter DataFrame
        """
        self.logger.info("Aggregating shooter data by Incident_ID")

        try:
            # Get all columns except Incident_ID
            col_agg = shooter_df.columns[1:]

            # Aggregate using custom function
            shooter_agg = shooter_df.groupby("Incident_ID").agg(
                {col: self.aggregate_categorical for col in col_agg}
            )

            # Add count of shooters
            shooter_agg["No_of_Shooters"] = (
                shooter_df.groupby("Incident_ID").size()
            )

            shooter_agg = shooter_agg.reset_index()

            self.logger.info(
                f"Aggregated shooter data: {len(shooter_agg)} unique incidents"
            )
            return shooter_agg

        except Exception as e:
            error_msg = f"Error aggregating shooter data: {str(e)}"
            self.logger.error(error_msg)
            raise

    def aggregate_victim_data(self, victim_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate victim data by Incident_ID.

        Args:
            victim_df: Victim DataFrame

        Returns:
            Aggregated victim DataFrame
        """
        self.logger.info("Aggregating victim data by Incident_ID")

        try:
            col_agg = victim_df.columns[1:]

            victim_agg = victim_df.groupby("Incident_ID").agg(
                {col: self.aggregate_categorical for col in col_agg}
            )

            # Add count of victims
            victim_agg["No_of_Victims"] = (
                victim_df.groupby("Incident_ID").size()
            )

            victim_agg = victim_agg.reset_index()

            self.logger.info(
                f"Aggregated victim data: {len(victim_agg)} unique incidents"
            )
            return victim_agg

        except Exception as e:
            error_msg = f"Error aggregating victim data: {str(e)}"
            self.logger.error(error_msg)
            raise

    def aggregate_weapon_data(self, weapon_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate weapon data by Incident_ID.

        Args:
            weapon_df: Weapon DataFrame

        Returns:
            Aggregated weapon DataFrame
        """
        self.logger.info("Aggregating weapon data by Incident_ID")

        try:
            col_agg = weapon_df.columns[1:]

            weapon_agg = weapon_df.groupby("Incident_ID").agg(
                {col: self.aggregate_categorical for col in col_agg}
            )

            weapon_agg = weapon_agg.reset_index()

            self.logger.info(
                f"Aggregated weapon data: {len(weapon_agg)} unique incidents"
            )
            return weapon_agg

        except Exception as e:
            error_msg = f"Error aggregating weapon data: {str(e)}"
            self.logger.error(error_msg)
            raise

    def merge_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all data sources into single DataFrame.

        Args:
            data: Dictionary with keys: incident, shooter, victim, weapon

        Returns:
            Merged DataFrame

        Raises:
            ValueError: If required data keys are missing
        """
        self.logger.info("Starting data merge process")

        required_keys = ["incident", "shooter", "victim", "weapon"]
        missing_keys = set(required_keys) - set(data.keys())

        if missing_keys:
            error_msg = f"Missing required data keys: {missing_keys}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Start with incident data
            df_merged = data["incident"].copy()
            self.logger.info(f"Starting with incident data: {len(df_merged)} rows")

            # Aggregate and merge shooter data
            shooter_agg = self.aggregate_shooter_data(data["shooter"])
            df_merged = df_merged.merge(shooter_agg, on="Incident_ID", how="left")
            self.logger.info(f"After shooter merge: {len(df_merged)} rows")

            # Aggregate and merge victim data
            victim_agg = self.aggregate_victim_data(data["victim"])
            df_merged = df_merged.merge(victim_agg, on="Incident_ID", how="left")
            self.logger.info(f"After victim merge: {len(df_merged)} rows")

            # Aggregate and merge weapon data
            weapon_agg = self.aggregate_weapon_data(data["weapon"])
            df_merged = df_merged.merge(weapon_agg, on="Incident_ID", how="left")
            self.logger.info(f"After weapon merge: {len(df_merged)} rows")

            self.logger.info(
                f"Merge complete: {len(df_merged)} rows, {len(df_merged.columns)} columns"
            )
            return df_merged

        except Exception as e:
            error_msg = f"Error merging data: {str(e)}"
            self.logger.error(error_msg)
            raise

    def load_and_merge(
        self, file_path: Optional[str] = None, save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Complete data loading and merging pipeline.

        Args:
            file_path: Path to Excel file
            save_csv: Whether to save merged data as CSV

        Returns:
            Merged DataFrame
        """
        self.logger.info("Starting data loading and merging pipeline")

        try:
            # Load Excel sheets
            data = self.load_excel_file(file_path)

            # Merge all data
            df_merged = self.merge_data(data)

            # Optionally save to CSV
            if save_csv:
                csv_path = self.config.get("data.output_csv")
                csv_path = Path(csv_path)
                csv_path.parent.mkdir(parents=True, exist_ok=True)

                df_merged.to_csv(csv_path, index=False)
                self.logger.info(f"Saved merged data to: {csv_path}")

            self.logger.info("Data loading and merging pipeline complete")
            return df_merged

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            raise

    def load_from_csv(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from pre-processed CSV file.

        Args:
            csv_path: Path to CSV file. If None, uses config default

        Returns:
            DataFrame

        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        if csv_path is None:
            csv_path = self.config.get("data.output_csv")

        csv_path = Path(csv_path)

        if not csv_path.exists():
            error_msg = f"CSV file not found: {csv_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info(f"Loading data from CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            error_msg = f"Error loading CSV: {str(e)}"
            self.logger.error(error_msg)
            raise
