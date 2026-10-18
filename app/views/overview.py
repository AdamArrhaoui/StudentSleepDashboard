# """
# Overview page: dataset size, a preview table, summary statistics and data quality.

# Drafted with help from Claude (Anthropic).
# """

# """
# Import statements: import necessary libraries and modules for the overview page.

# """
import plotly.express as px
import streamlit as st
from components import page_header, section_header
from data_loading import COLUMN_DESCRIPTIONS, TARGET, friendly_name, load_clean_data, load_raw_data


# """
# Build page header: display the title and description of the overview page.
# """

page_header(
    "Overview",
    "A first look at the student sleep dataset: how big it is, what it contains, "
    "and how it was cleaned before modelling.",
    isCaption=False
)

# Load data: load the raw and cleaned datasets, and identify numeric and categorical columns.
raw_df = load_raw_data()
df, cleaning_log = load_clean_data()
# Identify numeric and categorical columns for later use in summary statistics and data quality checks.
numeric_cols = df.select_dtypes(include="number").columns
categorical_cols = df.select_dtypes(exclude="number").columns


# Key numbers: a row of four metric cards - used for displaying widgets
col1, col2, col3, col4 = st.columns(4)

# """
# Set up metric cards: display key metrics about the dataset, including the number of students, features, average CGPA, and average weekday sleep.
# icon: use Material Design icons to visually represent each metric.
# delta: show the change in the number of students after cleaning the dataset.
# delta_color: set the color of the delta to grey, indicating that removing rows is neither good nor bad.
# delta_arrow: disable the arrow for the delta, as it is not relevant in this context.
# border: add a border around each metric card for visual separation.
# height: set the height of each metric card to 130 pixels for consistency for all cards.
# """

col1.metric(
    "Students",
    f"{len(df):,}",
    icon=":material/groups:",
    delta=f"{len(raw_df) - len(df)} removed in cleaning",
    delta_color="off",  # grey, no arrow: removing rows is neither good nor bad
    delta_arrow="off",
    border=True,
    height=130
)
col2.metric(
    "Features",
    len(df.columns) - 1,  # every column except the target
    icon=":material/view_column:",
    help="Columns used to predict CGPA",
    border=True,
    height=130
)
# For the number calculated in col3, we access the dataframe column TARGET and get the mean, and we use the format string to display it with two decimal places. The icon is set to a school icon, and the border and height are set for visual consistency.
col3.metric(
    "Average CGPA",
    f"{df[TARGET].mean():.2f} / 4.0",
    icon=":material/school:",
    border=True,
    height=130
)
col4.metric(
    "Average weekday sleep",
    f"{df['avg_sleep_hours_weekday'].mean():.1f} h",
    icon=":material/bedtime:",
    border=True,
    height=130
)

st.write("")  # a little breathing room before the tabs


# Details, split into tabs so the page stays short

# Unpacking the tabs: create four tabs for data preview, summary statistics, data quality, and target variable analysis.
# tab_preview: displays a preview of the cleaned dataset, allowing users to view a specified number of
# tab_stats: shows summary statistics for numeric and categorical columns, providing insights into the distribution and characteristics of the data.
# tab_quality: presents data quality information, including cleaning steps and missing values in the raw data,
# tab_target: focuses on the target variable (CGPA), displaying its distribution and key statistics.


tab_preview, tab_stats, tab_quality, tab_target = st.tabs(
    [
        ":material/table_rows: Data preview",
        ":material/functions: Summary statistics",
        ":material/fact_check: Data quality",
        ":material/school: Target: CGPA",
    ]
)


# With tab_preview: display the cleaned dataset preview, allowing users to select the number of rows to show and providing a column guide for better understanding of the data.
# Section header: "Cleaned data" - indicates that the following content relates to the cleaned dataset.
# number_of_rows: a slider that allows users to select the number of rows to display in the data preview, with a minimum of 5, maximum of 50, default value of 10, and step size of 5.
# st.dataframe: displays the cleaned dataset in a scrollable table format, with the specified number of rows and column configurations that include friendly names and tooltips for better understanding.
# st.caption: provides additional information about the data preview, including the number of rows shown and instructions for accessing the column guide.
# expander: "Column guide" - allows users to expand and view a guide that explains the meaning of each column in the dataset, making it easier to interpret the data.

with tab_preview:
    section_header("Cleaned data")
    number_of_rows = st.slider("Rows to show", min_value=5, max_value=50, value=10, step=5)
    st.dataframe(
        df.head(number_of_rows),
        hide_index=True,
        width="stretch",
        # Hovering a column name shows what it means
        column_config={
            col: st.column_config.Column(friendly_name(col), help=COLUMN_DESCRIPTIONS.get(col))
            for col in df.columns
        },
    )
    st.caption(
        f"Showing {number_of_rows} of {len(df):,} rows. Hover over a column name to see what it means, "
        "or open the column guide below."
    )

    # Same descriptions as the tooltips, but readable without a mouse
    with st.expander("Column guide"):
        st.dataframe(
            {
                "Column": [friendly_name(c) for c in df.columns],
                "Meaning": [COLUMN_DESCRIPTIONS.get(c, "") for c in df.columns],
            },
            hide_index=True,
            width="stretch",
        )

# 


# Summary Statistics tab: display information about the data cleaning process and missing values in the raw dataset, helping users understand how the data was processed and any potential issues with data quality.
with tab_stats:
    # Define the section header
    section_header("Numeric columns")
    # Variable numeric_summary: calculate summary statistics for numeric columns in the cleaned dataset, including count, mean, standard deviation, minimum, maximum, and percentiles. The index is set to friendly names for better readability, and the column names are renamed to more user-friendly labels.
    numeric_summary = df[numeric_cols].describe().T.round(2)
    # Set the index of numeric_summary to friendly names for better readability, and rename the columns to more user-friendly labels.
    numeric_summary.index = [friendly_name(c) for c in numeric_summary.index]
    # Rename the columns of numeric_summary to more user-friendly labels, such as "Count", "Mean", "Std. dev.", "Min", "25th pct.", "Median", "75th pct.", and "Max". This makes it easier for users to understand the summary statistics presented in the table.
    numeric_summary = numeric_summary.rename(
        columns={"count": "Count", "mean": "Mean", "std": "Std. dev.", "min": "Min",
                 "25%": "25th pct.", "50%": "Median", "75%": "75th pct.", "max": "Max"}
    )
    # Display the numeric_summary DataFrame in a scrollable table format, with the specified width and without the index. This allows users to easily view and analyze the summary statistics for numeric columns in the cleaned dataset.
    st.dataframe(numeric_summary, width="stretch")

    # Set the section header for categorical columns, and calculate summary statistics for categorical columns in the cleaned dataset. The index is set to friendly names for better readability, and the column names are renamed to more user-friendly labels. Finally, display the categorical_summary DataFrame in a scrollable table format, with the specified width and without the index.
    section_header("Categorical columns")
    # Variable categorical_summary: calculate summary statistics for categorical columns in the cleaned dataset, including count, unique values, most common value, and frequency of the most common value. The index is set to friendly names for better readability, and the column names are renamed to more user-friendly labels.
    categorical_summary = df[categorical_cols].describe().T
    # Set the index of categorical_summary to friendly names for better readability, and rename the columns to more user-friendly labels.
    categorical_summary.index = [friendly_name(c) for c in categorical_summary.index]
    # Rename the columns of categorical_summary to more user-friendly labels, such as "Count", "Categories", "Most common", and "Times seen". This makes it easier for users to understand the summary statistics presented in the table.
    categorical_summary = categorical_summary.rename(
        columns={"count": "Count", "unique": "Categories", "top": "Most common", "freq": "Times seen"}
    )
    # Display the categorical_summary DataFrame in a scrollable table format, with the specified width and without the index. This allows users to easily view and analyze the summary statistics for categorical columns in the cleaned dataset.
    st.dataframe(categorical_summary, width="stretch")

# Data Quality tab: display information about the data cleaning process and missing values in the raw dataset, helping users understand how the data was processed and any potential issues with data quality.
with tab_quality:
    # Define the section header for data quality, and create two columns to display the cleaning steps and missing values in the raw dataset. The left column will show a table of cleaning steps taken during data preprocessing, while the right column will display a bar chart of missing values per column in the raw dataset. This layout allows users to easily compare the cleaning process with the original data quality.
    left, right = st.columns(2, gap="large")

    # With left column: display the cleaning steps taken during data preprocessing, including the number of rows remaining after each step. This information helps users understand how the dataset was cleaned and what actions were taken to improve data quality.
    with left:
        # Section header: "Cleaning steps" - indicates that the following content relates to the data cleaning process.
        section_header("Cleaning steps")
        # Display the cleaning_log DataFrame in a scrollable table format, with the specified width and without the index. This allows users to easily view and analyze the steps taken during data preprocessing and the number of rows remaining after each step.
        st.dataframe(cleaning_log, hide_index=True, width="stretch")
        # Set the caption for the cleaning steps table, providing additional context about the data cleaning process. It explains that the table shows the number of rows remaining after each step of the cleaning pipeline and mentions that certain columns (Student ID and gender) were dropped before these steps.
        st.caption(
            "Rows remaining after each step of the cleaning pipeline. "
            "Student ID and gender were dropped as columns before these steps."
        )
    # With right column: display a bar chart of missing values per column in the raw dataset, helping users visualize the extent of missing data and identify potential issues with data quality. The chart shows the number of missing values for each column, allowing users to quickly assess which columns may require further attention or imputation.
    with right:
        # Section header: "Missing values in the raw data" - indicates that the following content relates to missing values in the original dataset.
        section_header("Missing values in the raw data")
        # Variable missing: calculate the number of missing values for each column in the raw dataset, and filter to only include columns with at least one missing value. The resulting Series is sorted in ascending order to make it easier to visualize the extent of missing data across different columns.
        missing = raw_df.isna().sum()
        # Filter to only include columns with at least one missing value, and sort the resulting Series in ascending order. This allows users to focus on columns that have missing data and assess the extent of the issue.
        missing = missing[missing > 0].sort_values()
        # Create a horizontal bar chart using Plotly Express to visualize the number of missing values per column in the raw dataset. The x-axis represents the number of missing values, while the y-axis shows the column names (with friendly names for better readability). The chart includes text labels displaying the count of missing values on each bar, and it is titled "Missing values per column (raw data)" with appropriate axis labels.
        fig = px.bar(
            x=missing.values,
            y=[friendly_name(c) for c in missing.index],
            orientation="h",
            text=missing.values,  # print the count on each bar, so colour isn't needed to read it
            title="Missing values per column (raw data)",
            labels={"x": "Number of missing values", "y": "Column"},
        )
        # Update the layout of the bar chart to set the y-axis title to "Column" and adjust the bar gap for better visual separation between bars.
        st.plotly_chart(fig, width="stretch")
        # Set the caption for the missing values bar chart, providing additional context about the missing data in the raw dataset. It explains how many rows had at least one missing value and that those rows were dropped rather than filled in, helping users understand the impact of missing data on the dataset.
        st.caption(
            f"{raw_df.isna().any(axis=1).sum()} of {len(raw_df)} raw rows had at least one "
            "missing value. Those rows were dropped rather than filled in."
        )

# Target variable distribution tab: display information about the target variable (CGPA), including its distribution and key statistics. This helps users understand the characteristics of the target variable and how it may impact model performance.
with tab_target:
    # Section header: "What we predict: CGPA" - indicates that the following content relates to the target variable (CGPA) and its distribution.
    section_header("What we predict: CGPA")
    # Create a histogram using Plotly Express to visualize the distribution of the target variable (CGPA) in the cleaned dataset. The x-axis represents CGPA values, while the y-axis shows the number of students. The histogram includes 25 bins, and it is titled "Distribution of student CGPA" with appropriate axis labels.
    fig = px.histogram(
        df,
        x=TARGET,
        nbins=25,
        title="Distribution of student CGPA",
        labels={TARGET: "CGPA (0-4 scale)", "count": "Number of students"},
    )
    # Update the layout of the histogram to set the y-axis title to "Number of students" and adjust the bar gap for better visual separation between bars.
    fig.update_layout(yaxis_title="Number of students", bargap=0.05)
    st.plotly_chart(fig, width="stretch")
    # Set the caption for the CGPA distribution histogram, providing additional context about the target variable. It explains the range of CGPA values, the median, and the interquartile range (middle 50% of students), helping users understand the distribution and characteristics of the target variable.
    # Format the CGPA values to two decimal places for better readability, and use the COLUMN_DESCRIPTIONS dictionary to provide additional information about the CGPA column if available.
    st.caption(
        f"CGPA ranges from {df[TARGET].min():.2f} to {df[TARGET].max():.2f}, "
        f"with a median of {df[TARGET].median():.2f}. Half of all students fall between "
        f"{df[TARGET].quantile(0.25):.2f} and {df[TARGET].quantile(0.75):.2f}."
    )
