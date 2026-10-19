# Loads the dataset for the dashboard pages.

# The Overview and Explore pages show data in real units (hours, ages, text
# categories), so they use the cleaned data *before* scaling and one-hot encoding.
# The model pages will use the processed files in data/processed/ instead.

# Drafted with help from Claude (Anthropic). 

# Import pathlib:- used to handle file paths in a platform-independent way. It provides an object-oriented interface for working with file system paths.
from pathlib import Path
# Import pandas as pd:- used for data manipulation and analysis. It provides data structures like DataFrame and Series, which are essential for handling tabular data.
import pandas as pd
# Import streamlit as st:- used to create interactive web applications for data science and machine learning. It allows for easy creation of UI elements like sliders, buttons, and tables.
import streamlit as st

# Import DataProcessor from src.data_processor:- imports the DataProcessor class from the data_processor module located in the src directory. This class is responsible for processing and cleaning the raw dataset, including tasks like removing duplicates, handling missing values, and removing outliers.
from src.data_processor import DataProcessor

# Paths are built from this file's location, so they work on any machine and in Docker.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Raw data path: points to the original CSV file containing the student sleep, screen time, and academic performance data. This path is constructed relative to the project root directory, ensuring that it works across different environments and machines.
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_sleep_screen_time_academic_performance.csv"

# Constant for the target variable (the value we predict) and its plain-language description. This is used in multiple places, so we define it here to avoid repeating it.
TARGET = "cgpa"

# Plain-language meaning of each column, shown as tooltips and in the column guide.
COLUMN_DESCRIPTIONS = {
    "age": "Student's age in years",
    "year_of_study": "Current year of study (1-5)",
    "major_category": "Field of study",
    "residence_type": "Where the student lives",
    "chronotype": "Natural sleep preference: early bird, intermediate or night owl",
    "avg_bedtime_hour": "Average bedtime on a 0-24 clock (1.5 = 1:30 AM)",
    "avg_sleep_hours_weekday": "Average hours of sleep on weeknights",
    "weekend_sleep_delta_hours": "Extra hours slept on weekends compared to weekdays",
    "pre_sleep_screen_time_min": "Minutes of screen use right before sleep",
    "daily_screen_time_hours": "Total screen time per day, in hours",
    "primary_device": "Device used most often",
    "device_free_before_bed": "Whether the student avoids devices before bed",
    "caffeine_drinks_per_day": "Caffeinated drinks per day",
    "naps_per_week": "Number of naps per week",
    "sleep_quality_score": "Self-rated sleep quality (0-10)",
    "exercise_days_per_week": "Days with exercise per week (0-7)",
    "energy_level": "Self-rated daytime energy (0-10)",
    "concentration_difficulty": "Self-rated difficulty concentrating (0-10, higher = harder)",
    "weekly_study_hours": "Hours spent studying per week",
    "class_attendance_percent": "Percentage of classes attended",
    "has_part_time_job": "Whether the student has a part-time job",
    "semester_type": "Semester when the data was collected",
    "cgpa": "Cumulative GPA on a 0-4 scale (the value we predict)",
}

# A function that takes a name of a column and returns a more human-readable version of it. For example, it turns "avg_sleep_hours_weekday" into "Avg sleep hours weekday". This is useful for displaying column names in the dashboard in a way that is easier for users to understand.
def friendly_name(column: str) -> str:
    """Turn a column name like 'avg_sleep_hours_weekday' into 'Avg sleep hours weekday'."""
    if column == "cgpa":
        return "CGPA"  # an acronym, so keep it in capitals
    # Return the column name with underscores replaced by spaces and the first letter capitalized. For example, "avg_sleep_hours_weekday" becomes "Avg sleep hours weekday".
    return column.replace("_", " ").capitalize()

# @st.cache_data is a decorator provided by Streamlit that caches the output of the function it decorates. When the function is called with the same input parameters, Streamlit will return the cached result instead of executing the function again. This is useful for expensive operations like loading data from disk, as it can significantly speed up the app by avoiding repeated computations. 
# So basically, it means that the function's result will be stored in memory, and if the function is called again with the same arguments, Streamlit will use the cached result instead of running the function again. This improves performance and reduces load times for the dashboard.
# take the raw data and return it as a pandas DataFrame. The data is read from the CSV file specified by RAW_DATA_PATH. This function is cached, so the data will only be loaded once per session, improving performance.
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """The original Kaggle CSV, exactly as downloaded."""
    # used the read_csv function from the pandas library, used the constant RAW_DATA_PATH to specify the path to the CSV file, and returned the resulting DataFrame. The function is decorated with @st.cache_data to cache the result for performance.
    return pd.read_csv(RAW_DATA_PATH)

# Load clean data will return a tuple of two pandas DataFrames: the cleaned data and a log of how many rows remained after each cleaning step. The cleaning steps include dropping unnecessary columns, removing duplicates, handling missing values, and removing outliers. The function is cached to improve performance, so the cleaning process will only be performed once per session.
@st.cache_data
def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same cleaning steps as run_processing_pipeline(), without scaling or encoding.

    Returns the cleaned data and a log of how many rows remained after each step.
    """
    # Processer variable is an instance of the DataProcessor class, which is initialized with the path to the raw data CSV file. The steps list is initialized with a tuple containing the description "Raw data" and the number of rows in the raw data. This will be used to keep track of how many rows remain after each cleaning step.
    # DataProcesser is also used for linear regression and random forest models, so it is imported from src.data_processor. The DataProcessor class contains methods for cleaning and processing the data, such as removing duplicates, handling missing values, and removing outliers.
    processor = DataProcessor(str(RAW_DATA_PATH))
    # Steps is a list of tuples that will be used to log the cleaning steps and the number of rows remaining after each step. The first tuple in the list contains the description "Raw data" and the number of rows in the raw data, which is obtained by accessing the length of the DataFrame stored in processor.data.
    steps = [("Raw data", len(processor.data))]

    # Drop unnecessary columns, remove duplicates, handle missing values, and remove outliers. After each step, append a tuple to the steps list containing a description of the step and the number of rows remaining in the data. Finally, create a DataFrame from the steps list to log the cleaning process and return the cleaned data and the cleaning log.
    processor.drop_columns(["student_id", "gender"])
    processor.remove_duplicates()
    # Append a tuple to the steps list containing the description "Removed duplicate rows" and the number of rows remaining in the data after removing duplicates. This is done by calling the remove_duplicates() method of the DataProcessor class, which removes any duplicate rows from the dataset. After this step, the length of the data is recorded in the steps list to keep track of how many rows remain after each cleaning step.
    steps.append(("Removed duplicate rows", len(processor.data)))

    # Handle missing values: remove rows with any missing values. This is done by calling the handle_missing() method of the DataProcessor class, which removes any rows that contain NaN (Not a Number) values. After this step, append a tuple to the steps list containing the description "Removed rows with missing values" and the number of rows remaining in the data.
    processor.handle_missing()
    # Append a tuple to the steps list containing the description "Removed rows with missing values" and the number of rows remaining in the data after handling missing values. This is done by calling the handle_missing() method of the DataProcessor class, which removes any rows that contain NaN (Not a Number) values. After this step, the length of the data is recorded in the steps list to keep track of how many rows remain after each cleaning step.
    steps.append(("Removed rows with missing values", len(processor.data)))

    # Remove outliers: remove rows where the z-score of any numeric column is greater than or equal to 3. This is done by calling the remove_outliers() method of the DataProcessor class, which calculates the z-score for each numeric column and removes any rows that have a z-score greater than or equal to 3. After this step, append a tuple to the steps list containing the description "Removed outliers (z-score ≥ 3)" and the number of rows remaining in the data.
    processor.remove_outliers(3.0)
    # Append a tuple to the steps list containing the description "Removed outliers (z-score ≥ 3)" and the number of rows remaining in the data after removing outliers. This is done by calling the remove_outliers() method of the DataProcessor class, which calculates the z-score for each numeric column and removes any rows that have a z-score greater than or equal to 3. After this step, the length of the data is recorded in the steps list to keep track of how many rows remain after each cleaning step.
    steps.append(("Removed outliers (z-score ≥ 3)", len(processor.data)))

    # Cleaning log is a DataFrame that contains the steps taken to clean the data and the number of rows remaining after each step. It is created from the steps list, which contains tuples of the step description and the number of rows remaining. The DataFrame has two columns: "Step" and "Rows remaining". Finally, the function returns a tuple containing the cleaned data (with the index reset) and the cleaning log.
    cleaning_log = pd.DataFrame(steps, columns=["Step", "Rows remaining"])
    # return the cleaned data (with the index reset) and the cleaning log. The cleaned data is obtained from the DataProcessor instance, and the index is reset to ensure that the DataFrame has a continuous index starting from 0. The cleaning log provides a summary of the cleaning steps and the number of rows remaining after each step, which can be useful for understanding how much data was removed during the cleaning process.
    # Index reset is important because after removing rows (duplicates, missing values, outliers), the original index may have gaps. Resetting the index ensures that the DataFrame has a continuous index starting from 0, which is often more convenient for further analysis and visualization.
    return processor.data.reset_index(drop=True), cleaning_log
