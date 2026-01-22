"""
Data Preprocessing Module
Comprehensive data cleaning and transformation pipeline.
Implements all preprocessing steps from the original notebook.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from ..utils.logger import get_logger
from ..utils.config import get_config


class DataPreprocessor:
    """Preprocess school shooting incident data."""

    def __init__(self):
        """Initialize DataPreprocessor."""
        self.config = get_config()
        self.logger = get_logger(__name__)

    # ============= SHOOTER DATA PREPROCESSING =============

    @staticmethod
    def categorize_age(age) -> str:
        """Categorize age into groups."""
        if pd.isna(age) or (isinstance(age, str) and age not in ['Teen', 'Adult']):
            return 'Unknown'
        elif age == 'Teen' or (isinstance(age, (int, float)) and 13 <= age <= 19):
            return 'Teen'
        elif age == 'Adult' or (isinstance(age, (int, float)) and 20 <= age < 60):
            return 'Adult'
        elif isinstance(age, (int, float)) and age >= 60:
            return 'Old_Age'
        else:
            return 'Unknown'

    @staticmethod
    def categorize_school_affiliation(affiliation) -> str:
        """Categorize school affiliation."""
        if pd.isna(affiliation):
            return 'Unknown'

        if affiliation in ['Student', 'Former Student', 'Other Student', 'Rival School Student']:
            return 'Student/Former_Student'
        elif affiliation in ['Teacher', 'Principal/Vice-Principal', 'Former Teacher', 'Other Staff']:
            return 'Staff/Former_Staff'
        elif affiliation in ['Parent', 'Relative', 'Intimate Relationship', 'Friend']:
            return 'Family/Intimate'
        elif affiliation in ['No Relation', 'Nonstudent', 'Nonstudent Using Athletic Facilities/Attending Game', 'Gang Member', 'Hitman']:
            return 'No_School_Relation'
        elif affiliation in ['Security Guard', 'Police Officer/SRO']:
            return 'Law_Enforcement'
        elif affiliation == 'Unknown':
            return 'Unknown'
        else:
            return 'Other'

    @staticmethod
    def categorize_shooter_outcome(outcome) -> str:
        """Categorize shooter outcome."""
        if pd.isna(outcome):
            return 'Other/Unknown'

        if outcome in ['Fled/Apprehended', 'Fled/Escaped']:
            return 'Fled'
        elif outcome in ['Apprehended/Killed by LE', 'Apprehended/Killed by SRO', 'Apprehended/Killed by Other']:
            return 'Apprehended/Killed'
        elif outcome in ['Suicide', 'Attempted Suicide']:
            return 'Suicide/Attempted Suicide'
        elif outcome == 'Subdued by Students/Staff/Other':
            return 'Subdued'
        elif outcome == 'Surrendered':
            return 'Surrendered'
        elif outcome == 'Law Enforcement':
            return 'Law Enforcement'
        elif outcome in ['Unknown', 'Other', None]:
            return 'Other/Unknown'
        else:
            return outcome

    def preprocess_shooter_data(self, shooter_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess shooter data with all transformations.

        Args:
            shooter_df: Raw shooter DataFrame

        Returns:
            Preprocessed shooter DataFrame
        """
        self.logger.info("Preprocessing shooter data")
        df = shooter_df.copy()

        # Rename columns
        df = df.rename(columns={
            'Age': 'Shooter_Age',
            'Gender': 'Shooter_Gender',
            'Race': 'Shooter_Race',
            'School_Affiliation': 'Shooter_School_Affiliation',
            'Injury': 'Shooter_Injury'
        })

        # Categorize age
        df['Shooter_Age'] = df['Shooter_Age'].apply(self.categorize_age)

        # Gender cleaning
        df['Shooter_Gender'] = df['Shooter_Gender'].fillna('Unknown')
        df['Shooter_Gender'] = df['Shooter_Gender'].replace({
            'Transgender': 'Other',
            'Multiple': 'Other'
        })

        # Race grouping
        df['Shooter_Race'] = df['Shooter_Race'].replace({
            'Other': 'Other/Minority',
            'Native American/Alaska Native': 'Other/Minority',
            'Hawaiian/Pacific Islander': 'Other/Minority',
            'Middle Eastern': 'Other/Minority'
        })
        df['Shooter_Race'] = df['Shooter_Race'].fillna('Unknown')

        # School affiliation
        df['Shooter_School_Affiliation'] = df['Shooter_School_Affiliation'].apply(
            self.categorize_school_affiliation
        )

        # Shooter outcome
        df['Shooter_Outcome'] = df['Shooter_Outcome'].apply(self.categorize_shooter_outcome)
        df['Shooter_Outcome'] = df['Shooter_Outcome'].fillna('Other/Unknown')

        # Shooter died and injury
        df['Shooter_Died'] = df['Shooter_Died'].fillna('Unknown')
        df['Shooter_Injury'] = df['Shooter_Injury'].fillna('Unknown')
        df['Shooter_Injury'] = df['Shooter_Injury'].replace('None', 'No_Injury')

        self.logger.info("Shooter data preprocessing complete")
        return df

    # ============= VICTIM DATA PREPROCESSING =============

    @staticmethod
    def categorize_victim_gender(gender) -> str:
        """Categorize victim gender."""
        if gender == 'Male':
            return 'Male'
        elif gender == 'Female':
            return 'Female'
        else:
            return 'Unknown'

    def preprocess_victim_data(self, victim_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess victim data with all transformations.

        Args:
            victim_df: Raw victim DataFrame

        Returns:
            Preprocessed victim DataFrame
        """
        self.logger.info("Preprocessing victim data")
        df = victim_df.copy()

        # Rename columns
        df = df.rename(columns={
            'Injury': 'Victim_Injury',
            'Gender': 'Victim_Gender',
            'School_Affiliation': 'Victim_School_Affiliation',
            'Age': 'Victim_Age',
            'Race': 'Victim_Race'
        })

        # Victim injury
        df['Victim_Injury'] = df['Victim_Injury'].replace('None', 'No_Injury')

        # Victim gender
        df['Victim_Gender'] = df['Victim_Gender'].apply(self.categorize_victim_gender)

        # Victim school affiliation
        df['Victim_School_Affiliation'] = df['Victim_School_Affiliation'].apply(
            self.categorize_school_affiliation
        )

        # Victim age
        df['Victim_Age'] = df['Victim_Age'].apply(self.categorize_age)

        # Victim race
        df['Victim_Race'] = df['Victim_Race'].fillna('Unknown')

        self.logger.info("Victim data preprocessing complete")
        return df

    # ============= WEAPON DATA PREPROCESSING =============

    @staticmethod
    def categorize_weapon(weapon) -> str:
        """Categorize weapon type."""
        if pd.isna(weapon):
            return 'Unknown/Not_Specified'

        if weapon in ['Handgun', 'Rifle', 'Shotgun']:
            return weapon
        elif weapon in ['Multiple Handguns', 'Multiple Rifles', 'Multiple Unknown']:
            return 'Multiple_Weapons'
        elif weapon in ['No Data', 'NaN', 'Unknown', None]:
            return 'Unknown/Not_Specified'
        elif weapon == 'Other':
            return 'Other'
        else:
            return 'Unknown/Not_Specified'

    def preprocess_weapon_data(self, weapon_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess weapon data with all transformations.

        Args:
            weapon_df: Raw weapon DataFrame

        Returns:
            Preprocessed weapon DataFrame
        """
        self.logger.info("Preprocessing weapon data")
        df = weapon_df.copy()

        # Drop columns
        columns_to_drop = ['Weapon_Caliber', 'Weapon_Details']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

        # Categorize weapon type
        df['Weapon_Type'] = df['Weapon_Type'].apply(self.categorize_weapon)

        self.logger.info("Weapon data preprocessing complete")
        return df

    # ============= MERGED DATA PREPROCESSING =============

    @staticmethod
    def categorize_targets(value) -> str:
        """Categorize targets."""
        if pd.isna(value):
            return 'Unknown'
        elif value in ['Both', 'Neither']:
            return 'Mixed/Unclear'
        elif value == 'Victims Targeted':
            return 'Victims_Targeted'
        elif value == 'Random Shooting':
            return 'Random_Shooting'
        else:
            return value

    @staticmethod
    def categorize_shooters_count(value) -> str:
        """Categorize number of shooters."""
        if pd.isna(value):
            return 'Unknown'
        elif value == 1.0 or value == 1:
            return 'Single_Shooter'
        else:
            return 'Multiple_Shooters'

    @staticmethod
    def categorize_situation(situation) -> str:
        """Categorize situation type."""
        if pd.isnull(situation):
            return 'Unknown'
        if situation in ['Escalation of Dispute', 'Anger Over Grade/Suspension/Discipline', 'Bullying', 'Racial']:
            return 'Conflict_Related'
        elif situation in ['Illegal Activity', 'Drive-by Shooting', 'Intentional Property Damage']:
            return 'Criminal_Activity'
        elif situation in ['Suicide/Attempted', 'Domestic w/ Targeted Victim', 'Psychosis']:
            return 'Personal_Crisis'
        elif situation in ['Officer-Involved Shooting', 'Hostage/Standoff']:
            return 'Law_Enforcement'
        elif situation == 'Accidental':
            return 'Accidental'
        else:
            return 'Other'

    @staticmethod
    def categorize_time_period(time_period) -> str:
        """Categorize time period."""
        if pd.isnull(time_period) or time_period == 'Unknown':
            return 'Unknown'
        if time_period in ['Morning Classes', 'School Start']:
            return 'During_Morning_Hours'
        elif time_period in ['Afternoon Classes', 'Lunch']:
            return 'During_Afternoon_Hours'
        elif time_period in ['After School', 'Dismissal']:
            return 'After_School_Hours'
        elif time_period in ['Sport Event', 'School Event']:
            return 'School_Related_Event'
        elif time_period in ['Evening', 'Night']:
            return 'Evening/Night'
        elif time_period in ['Not a School Day', 'Not A School Day']:
            return 'Non_School Day'
        else:
            return 'Other'

    @staticmethod
    def categorize_location(location) -> str:
        """Categorize location."""
        if pd.isnull(location) or location in ['ND', 'Other', 'NaN']:
            return 'Unknown'
        if any(sports_related in location for sports_related in ['Football Field', 'Track', 'Gym', 'Basketball Court', 'Playground', 'Field']):
            return 'Sports/Gym'
        if 'Office' in location:
            return 'Office'
        if any(indoor in location for indoor in ['Classroom', 'Hallway', 'Bathroom', 'Cafeteria', 'Auditorium', 'Inside', 'Inside School Building']):
            return 'Inside_School_Building'
        if 'Parking Lot' in location:
            return 'Parking_Lot'
        if any(outside in location for outside in ['Front of School', 'Beside Building', 'Outside', 'Courtyard', 'Off School', 'Outside on School Property']):
            return 'Outside'
        if 'Entryway' in location:
            return 'Inside_School_Building'
        if 'School Bus' in location:
            return 'School_Bus'
        return location

    @staticmethod
    def categorize_time_of_day(time_value) -> str:
        """Categorize time of day from datetime."""
        if pd.isna(time_value):
            return 'Unknown'
        try:
            if hasattr(time_value, 'hour'):
                hour = time_value.hour
                if hour < 6:
                    return 'Night'
                elif 6 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 18:
                    return 'Afternoon'
                else:
                    return 'Evening'
        except:
            return 'Unknown'
        return 'Unknown'

    @staticmethod
    def categorize_school_level(level) -> str:
        """Categorize school level."""
        if pd.isna(level):
            return 'Other_Unknown'
        if level in ['Elementary', 'K-8']:
            return 'Primary_Education'
        elif level in ['Middle', 'Junior High', '6-12', 'K-12']:
            return 'Secondary_Education'
        elif level == 'High':
            return 'High_School'
        else:
            return 'Other_Unknown'

    @staticmethod
    def categorize_location_type(location_type) -> str:
        """Categorize location type."""
        if pd.isna(location_type):
            return 'Unknown_Other'
        if location_type in ['Outside on School Property', 'Inside School Building', 'School Bus']:
            return 'On_School_Property'
        elif location_type == 'Off School Property':
            return 'Off_School_Property'
        elif location_type == 'Both Inside/Outside':
            return 'Mixed_Location'
        else:
            return 'Unknown_Other'

    def preprocess_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess merged dataframe with all transformations.

        Args:
            df: Merged DataFrame

        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Preprocessing merged data")
        df = df.copy()

        # Drop columns from config
        drop_columns = self.config.get("preprocessing.drop_columns", [])
        existing_drop_cols = [col for col in drop_columns if col in df.columns]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols, axis=1)
            self.logger.info(f"Dropped {len(existing_drop_cols)} columns")

        # Fill missing values
        default_fill = self.config.get("preprocessing.default_categorical_fill", "Unknown")

        # Active Shooter FBI
        if 'Active_Shooter_FBI' in df.columns:
            df['Active_Shooter_FBI'] = df['Active_Shooter_FBI'].fillna(default_fill)

        # Media Attention
        if 'Media_Attention' in df.columns:
            df['Media_Attention'] = df['Media_Attention'].fillna(default_fill)

        # Involves Students/Staff
        if 'Involves_Students_Staff' in df.columns:
            df['Involves_Students_Staff'] = df['Involves_Students_Staff'].replace({'Y': 'Yes'})
            df['Involves_Students_Staff'] = df['Involves_Students_Staff'].fillna(default_fill)

        # Gang Related
        if 'Gang_Related' in df.columns:
            df['Gang_Related'] = df['Gang_Related'].fillna(default_fill)

        # Victim columns
        victim_cols = ['Victim_Injury', 'Victim_Race', 'Victim_Age', 'Victim_School_Affiliation', 'Victim_Gender']
        for col in victim_cols:
            if col in df.columns:
                if col == 'Victim_Injury':
                    df[col] = df[col].fillna('No_Injury')
                else:
                    df[col] = df[col].fillna(default_fill)

        # Accomplice
        if 'Accomplice' in df.columns:
            df['Accomplice'] = df['Accomplice'].fillna(default_fill)

        # Bullied
        if 'Bullied' in df.columns:
            df['Bullied'] = df['Bullied'].fillna(default_fill).replace({'N': 'No'})

        # Domestic Violence
        if 'Domestic_Violence' in df.columns:
            df['Domestic_Violence'] = df['Domestic_Violence'].fillna(default_fill).replace({'NO': 'No'})

        # Weapon Type
        if 'Weapon_Type' in df.columns:
            df['Weapon_Type'] = df['Weapon_Type'].fillna('Unknown/Not_Specified')

        # During Classes
        if 'During_Classes' in df.columns:
            df['During_Classes'] = df['During_Classes'].fillna(default_fill)

        # Hostages
        if 'Hostages' in df.columns:
            df['Hostages'] = df['Hostages'].fillna(default_fill)

        # Barricade
        if 'Barricade' in df.columns:
            df['Barricade'] = df['Barricade'].fillna(default_fill).replace({'N': 'No'})

        # Quarter
        if 'Quarter' in df.columns:
            df['Quarter'] = df['Quarter'].fillna(default_fill)

        # Officer Involved
        if 'Officer_Involved' in df.columns:
            df['Officer_Involved'] = df['Officer_Involved'].fillna(default_fill).replace({'N': 'No'})

        # Shooter columns
        shooter_cols = ['Shooter_Age', 'Shooter_Gender', 'Shooter_Race', 'Shooter_School_Affiliation', 'Shooter_Outcome', 'Shooter_Died', 'Shooter_Injury']
        for col in shooter_cols:
            if col in df.columns:
                df[col] = df[col].fillna(default_fill)

        # Duration - median imputation
        if 'Duration_min' in df.columns:
            imputation_method = self.config.get("preprocessing.duration_imputation_method", "median")
            if imputation_method == "median":
                median_duration = df['Duration_min'].median()
                df['Duration_min'].fillna(median_duration, inplace=True)
                self.logger.info(f"Filled Duration_min with median: {median_duration}")
            elif imputation_method == "mean":
                mean_duration = df['Duration_min'].mean()
                df['Duration_min'].fillna(mean_duration, inplace=True)
                self.logger.info(f"Filled Duration_min with mean: {mean_duration}")

        # Narrative
        if 'Narrative' in df.columns:
            df['Narrative'] = df['Narrative'].fillna('No Narrative')

        # Apply categorization functions
        if 'Targets' in df.columns:
            df['Targets'] = df['Targets'].apply(self.categorize_targets)

        if 'No_of_Shooters' in df.columns:
            df['No_of_Shooters'] = df['No_of_Shooters'].apply(self.categorize_shooters_count)

        if 'Situation' in df.columns:
            df['Situation'] = df['Situation'].apply(self.categorize_situation)

        if 'Time_Period' in df.columns:
            df['Time_Period'] = df['Time_Period'].apply(self.categorize_time_period)

        if 'Location' in df.columns:
            df['Location'] = df['Location'].apply(self.categorize_location)

        if 'First_Shot' in df.columns:
            df['First_Shot'] = df['First_Shot'].apply(self.categorize_time_of_day)

        if 'School_Level' in df.columns:
            df['School_Level'] = df['School_Level'].apply(self.categorize_school_level)

        if 'Location_Type' in df.columns:
            df['Location_Type'] = df['Location_Type'].apply(self.categorize_location_type)

        self.logger.info("Merged data preprocessing complete")
        return df

    def full_preprocessing_pipeline(
        self,
        shooter_df: pd.DataFrame,
        victim_df: pd.DataFrame,
        weapon_df: pd.DataFrame,
        merged_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Execute full preprocessing pipeline on all data.

        Args:
            shooter_df: Raw shooter data
            victim_df: Raw victim data
            weapon_df: Raw weapon data
            merged_df: Merged data

        Returns:
            Fully preprocessed DataFrame
        """
        self.logger.info("Starting full preprocessing pipeline")

        # Preprocess individual dataframes
        shooter_clean = self.preprocess_shooter_data(shooter_df)
        victim_clean = self.preprocess_victim_data(victim_df)
        weapon_clean = self.preprocess_weapon_data(weapon_df)

        # Preprocess merged data
        final_df = self.preprocess_merged_data(merged_df)

        self.logger.info("Full preprocessing pipeline complete")
        return final_df
