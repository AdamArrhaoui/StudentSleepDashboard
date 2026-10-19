"""
Explore page: interactive charts that respond to filters and dropdowns.

Drafted with help from Claude (Anthropic).
"""

# Import necessary libraries for data manipulation, visualization, and web app development
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import the custom components and data loading functions from the local modules
from components import page_header, section_header
# Import the COLUMN_DESCRIPTIONS, TARGET, friendly_name, and load_clean_data functions from the data_loading module to access the dataset and its metadata
from data_loading import COLUMN_DESCRIPTIONS, TARGET, friendly_name, load_clean_data

# Page header: "Explore" - indicates that the following content relates to exploring the dataset and its relationships.
# isCaption=False - indicates that the description should not be displayed as a caption, but rather as a main title for the page.
page_header(
    "Explore",
    "Filter the students and see how sleep, screens and lifestyle relate to CGPA.",
    isCaption=False,
)

# dataframe, _ = load_clean_data() - loads the cleaned dataset into a pandas DataFrame for further analysis and visualization. The underscore (_) is used to ignore any additional return values from the load_clean_data function that are not needed in this context.
# The _ is a way to discard the second return value from the load_clean_data function, which may contain metadata or other information that is not relevant for this page.
df, _ = load_clean_data()
# The cleaned dataset is now available in the df variable for further analysis and visualization on the Explore page.
numeric_cols = df.select_dtypes(include="number").columns.tolist()
# The numeric_cols variable contains a list of column names in the DataFrame that have numeric data types. This is useful for identifying which columns can be used for numerical analysis and visualizations, such as scatter plots, histograms, and correlation matrices.
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

# Plotly labels: show friendly names instead of raw column names on axes and legends
LABELS = {col: friendly_name(col) for col in df.columns}

# Filters: one row above the charts, applied to every chart on the page

# With st.container(border=True): - creates a container in the Streamlit app with a border around it. This container will hold the filter widgets that allow users to filter the dataset based on various criteria.
# Container is used to group related elements together and provide a visual separation from other parts of the page.
with st.container(border=True):
    # st.markdown("**:material/filter_alt: Filter students**") - adds a markdown header with an icon to indicate that the following widgets are for filtering the students in the dataset. The icon is a filter symbol, and the text is bolded for emphasis.
    st.markdown("**:material/filter_alt: Filter students**")
    # The following code creates four columns in the container, each containing a different filter widget for the user to interact with. These filters allow users to narrow down the dataset based on specific criteria, such as major, chronotype, residence type, and year of study.
    f1, f2, f3, f4 = st.columns(4) # Unpacking the four columns into variables f1, f2, f3, and f4 for easier reference when adding filter widgets to each column.
    # Majors is a variable where we use multiselect, multiselect allows the user to select multiple options from a list of majors. The options are sorted and the placeholder text is "All majors". If the user does not select any majors, it means no filter is applied for majors.
    # We use the sorted function to ensure that the list of majors is displayed in alphabetical order, making it easier for users to find and select their desired options.
    # f1 is the first column in the container, and we add the multiselect widget for majors to this column. The selected majors will be stored in the majors variable for later use when filtering the dataset.
    majors = f1.multiselect(
        "Major", sorted(df["major_category"].unique()), placeholder="All majors"
    )

    # Chronotypes is a variable where we use multiselect, multiselect allows the user to select multiple options from a list of chronotypes. The options are sorted and the placeholder text is "All chronotypes". If the user does not select any chronotypes, it means no filter is applied for chronotypes.
    chronotypes = f2.multiselect(
        "Chronotype", sorted(df["chronotype"].unique()), placeholder="All chronotypes"
    )
    # Residences is a variable where we use multiselect, multiselect allows the user to select multiple options from a list of residence types. The options are sorted and the placeholder text is "All residences". If the user does not select any residence types, it means no filter is applied for residences.
    residences = f3.multiselect(
        "Residence", sorted(df["residence_type"].unique()), placeholder="All residences"
    )
    # Min_year and max_year are variables that store the minimum and maximum values of the "year_of_study" column in the dataset. These values are used to create a slider widget that allows users to filter students based on their year of study.
    # The slider will have a range from min_year to max_year, and users can select a specific range of years to filter the dataset accordingly.
    # The f4 variable is the fourth column in the container, and we add the slider widget for year of study to this column. The selected range of years will be stored in the years variable for later use when filtering the dataset.
    # We are accessing the dataframe df and using the min() and max() functions to find the minimum and maximum values of the "year_of_study" column. We then convert these values to integers to ensure that the slider widget displays whole numbers for the year of study. And using int() to convert the min and max values to integers, we ensure that the slider widget displays whole numbers for the year of study, which is more user-friendly and appropriate for this context.
    min_year, max_year = int(df["year_of_study"].min()), int(df["year_of_study"].max())
    years = f4.slider("Year of study", min_year, max_year, (min_year, max_year))

# An empty multiselect means "no filter", so only filter when something is picked
filtered = df[df["year_of_study"].between(*years)]
# If the user has selected any majors, chronotypes, or residence types in the respective multiselect widgets, we filter the dataset accordingly. If no options are selected in a multiselect widget, it means no filter is applied for that criterion.
if majors:
    filtered = filtered[filtered["major_category"].isin(majors)]
if chronotypes:
    filtered = filtered[filtered["chronotype"].isin(chronotypes)]
if residences:
    filtered = filtered[filtered["residence_type"].isin(residences)]

# A caption is added to the page to inform users about the number of students that match the current filters. It shows the count of filtered students and the total number of students in the dataset. This provides users with immediate feedback on how their filter selections are affecting the dataset.
st.caption(f"Showing **{len(filtered)}** of {len(df)} students.")

# If the length of the filtered dataset is less than 5, a warning message is displayed to inform users that there are too few students matching the current filters to draw meaningful charts. The message suggests removing a filter to increase the number of matching students. The st.stop() function is called to halt the execution of the rest of the page, preventing any further visualizations or analyses from being displayed when there are insufficient data points.
if len(filtered) < 5:
    st.warning("Too few students match these filters to draw meaningful charts. Try removing a filter.")
    st.stop()  # stop running the rest of the page

# Charts, one per tab
# Unpacking the four tabs into variables tab_rel, tab_groups, tab_dist, and tab_corr for easier reference when adding content to each tab.
tab_rel, tab_groups, tab_dist, tab_corr = st.tabs(
    [
        ":material/scatter_plot: Relationships",
        ":material/candlestick_chart: Compare groups",
        ":material/bar_chart: Distributions",
        ":material/grid_on: Correlations",
    ]
)

# Feature columns are defined as all numeric columns in the dataset except for the target variable (CGPA). This list of feature columns will be used for creating visualizations and analyses that explore the relationships between different features and the target variable.
feature_cols = [c for c in numeric_cols if c != TARGET]

# With tab_rel: - creates a context for adding content to the "Relationships" tab. All Streamlit commands within this context will be applied to the "Relationships" tab, allowing for a clean and organized layout.
with tab_rel:
    # Section header: "How does one habit relate to another?" - indicates that the following content relates to exploring the relationships between different habits and their impact on CGPA.
    section_header("How does one habit relate to another?")
    # The following code creates three columns in the "Relationships" tab, each containing a different dropdown widget for selecting the X-axis, Y-axis, and color grouping for the scatter plot. These dropdowns allow users to customize the scatter plot based on their interests and explore the relationships between different features in the dataset.
    c1, c2, c3 = st.columns(3)
    # X_col is a variable that stores the selected feature for the X-axis of the scatter plot. The selectbox widget allows users to choose from the list of feature columns, with "avg_sleep_hours_weekday" set as the default selection. The friendly_name function is used to display user-friendly names for the features in the dropdown.
    # Select box is a Streamlit widget that creates a dropdown menu for users to select an option from a list. In this case, it allows users to choose which feature they want to display on the X-axis of the scatter plot.
    x_col = c1.selectbox(
        "X axis", feature_cols,
        # index=feature_cols.index("avg_sleep_hours_weekday") - sets the default selection for the X-axis to "avg_sleep_hours_weekday" by finding its index in the feature_cols list. This ensures that when the page loads, the scatter plot will initially display average sleep hours on weekdays as the X-axis.
        index=feature_cols.index("avg_sleep_hours_weekday"),
        # format_func=friendly_name, - uses the friendly_name function to display user-friendly names for the features in the dropdown menu. This makes it easier for users to understand the options available for selection, as the raw column names may not be as intuitive or descriptive.
        format_func=friendly_name,
    )
    # Y_col is a variable that stores the selected feature for the Y-axis of the scatter plot. The selectbox widget allows users to choose from the list of numeric columns, with the target variable (CGPA) set as the default selection. The friendly_name function is used to display user-friendly names for the features in the dropdown.
    y_col = c2.selectbox(
        "Y axis", numeric_cols, index=numeric_cols.index(TARGET), format_func=friendly_name
    )
    # Colour_col is a variable that stores the selected categorical feature for color grouping in the scatter plot. The selectbox widget allows users to choose from the list of categorical columns, with an additional option for "Nothing" (None) to indicate no color grouping. The friendly_name function is used to display user-friendly names for the features in the dropdown.
    colour_col = c3.selectbox(
        # "Colour by" - label for the dropdown menu that allows users to select a categorical feature for color grouping in the scatter plot.
        "Colour by", [None] + categorical_cols,
        # index=0 - sets the default selection for the color grouping to "Nothing" (None), meaning that initially, there will be no color grouping applied to the scatter plot. 
        # Lambda function is used to customize the display of the options in the dropdown menu. If the selected option is None, it will display "Nothing" in the dropdown; otherwise, it will use the friendly_name function to display a user-friendly name for the selected categorical feature.
        format_func=lambda c: "Nothing" if c is None else friendly_name(c),
    )

    # Fig is a variable that stores the scatter plot figure created using Plotly Express. The px.scatter function is used to create the scatter plot based on the filtered dataset and the selected X-axis, Y-axis, and color grouping features. The title and labels for the axes are also set using user-friendly names.
    fig = px.scatter(
        filtered,
        x=x_col,
        y=y_col,
        color=colour_col,
        symbol=colour_col,  # shapes as well as colours, so groups don't rely on colour alone
        opacity=0.8,
        title=f"{friendly_name(y_col)} vs. {friendly_name(x_col)}",
        labels=LABELS,
    )
    # Update the marker size for the scatter plot to make the points more visible and easier to distinguish. The update_traces function is used to modify the marker properties of the scatter plot, setting the size of the markers to 8.
    # update_traces is a method in Plotly that allows you to modify the properties of the traces (data series) in a figure. In this case, we are updating the marker size for the scatter plot to make the points more visible and easier to distinguish.
    fig.update_traces(marker={"size": 8})

    # Straight-line trend (least squares), drawn on top of the dots
    slope, intercept = np.polyfit(filtered[x_col], filtered[y_col], 1)
    x_line = np.array([filtered[x_col].min(), filtered[x_col].max()])
    fig.add_trace(
        go.Scatter(
            x=x_line, y=slope * x_line + intercept, mode="lines", name="Trend line",
            line={"width": 2, "dash": "dash", "color": "#F1DAC4"},
        )
    )
    # Add the scatter plot figure to the Streamlit app using st.plotly_chart. The width parameter is set to "stretch" to make the chart responsive and fill the available horizontal space in the app layout.
    st.plotly_chart(fig, width="stretch")

    # Calculate the correlation coefficient (r) between the selected X-axis and Y-axis features using the pandas corr() method. The strength and direction of the correlation are determined based on the value of r, and a caption is added to the page to inform users about the correlation between the two features, along with a brief description of the selected X-axis feature.
    r = filtered[x_col].corr(filtered[y_col])
    # Determine the strength of the correlation based on the absolute value of r. If the absolute value of r is less than 0.3, the correlation is considered "weak"; if it is between 0.3 and 0.6, it is considered "moderate"; and if it is greater than or equal to 0.6, it is considered "strong".
    strength = "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.6 else "strong"
    # Determine the direction of the correlation based on the sign of r. If r is greater than 0, the correlation is considered "positive"; if r is less than or equal to 0, it is considered "negative".
    direction = "positive" if r > 0 else "negative"
    # Add a caption to the page that displays the correlation coefficient (r), its strength, and direction, along with a brief description of the selected X-axis feature. The COLUMN_DESCRIPTIONS dictionary is used to provide additional context about the selected X-axis feature, if available.
    st.caption(
        f"Correlation r = {r:.2f}: a {strength} {direction} relationship. "
        f"{COLUMN_DESCRIPTIONS.get(x_col, '')}."
    )

# with tab_groups: - creates a context for adding content to the "Compare groups" tab. All Streamlit commands within this context will be applied to the "Compare groups" tab, allowing for a clean and organized layout.
with tab_groups:
    # Section header: "How do groups of students compare?" - indicates that the following content relates to comparing different groups of students based on selected features.
    section_header("How do groups of students compare?")
    # The following code creates two columns in the "Compare groups" tab, each containing a different dropdown widget for selecting the grouping feature and the value feature for the box plot. These dropdowns allow users to customize the box plot based on their interests and explore how different groups of students compare in terms of selected features.
    c1, c2 = st.columns(2)
    # group_col is a variable that stores the selected categorical feature for grouping students in the box plot. The selectbox widget allows users to choose from the list of categorical columns, and the friendly_name function is used to display user-friendly names for the features in the dropdown.
    group_col = c1.selectbox(
        "Group students by", categorical_cols, format_func=friendly_name
    )
    # value_col is a variable that stores the selected numeric feature for the Y-axis of the box plot. The selectbox widget allows users to choose from the list of numeric columns, with the target variable (CGPA) set as the default selection. The friendly_name function is used to display user-friendly names for the features in the dropdown.
    value_col = c2.selectbox(
        "Compare", numeric_cols, index=numeric_cols.index(TARGET), format_func=friendly_name
    )

    # Order the groups by their median, so the chart reads from lowest to highest
    order = filtered.groupby(group_col)[value_col].median().sort_values().index.tolist()
    # Fig is a variable that stores the box plot figure created using Plotly Express. The px.box function is used to create the box plot based on the filtered dataset and the selected grouping and value features. The title and labels for the axes are also set using user-friendly names.
    fig = px.box(
        filtered,
        x=group_col,
        y=value_col,
        points="outliers",
        category_orders={group_col: order},
        title=f"{friendly_name(value_col)} by {friendly_name(group_col).lower()}",
        labels=LABELS,
    )
    # Streamlit command to display the box plot figure in the "Compare groups" tab. The width parameter is set to "stretch" to make the chart responsive and fill the available horizontal space in the app layout.
    st.plotly_chart(fig, width="stretch")

    # Median values for each group are calculated using the pandas groupby() method, and a summary caption is added to the page to inform users about the highest and lowest median values among the groups. If there is only one group matching the current filters, a message is displayed indicating that only one group is available for comparison. The caption also provides information about how to interpret the box plot, including the middle 50% of students and the median line.
    medians = filtered.groupby(group_col)[value_col].median()
    # if len(medians) > 1: - checks if there is more than one group in the filtered dataset. If there are multiple groups, it calculates the highest and lowest median values and creates a summary caption. If there is only one group, it displays a message indicating that only one group is available for comparison.
    if len(medians) > 1:
        # Summary is the median values for each group, and it is used to create a summary caption that informs users about the highest and lowest median values among the groups. The idxmax() and idxmin() methods are used to find the group names corresponding to the highest and lowest median values, respectively. The max() and min() methods are used to find the actual median values for those groups.
        # medians.idxmax() - returns the index (group name) of the group with the highest median value for the selected feature. This is used to identify which group has the highest median in the summary caption.
        # medians.max() - returns the highest median value among all groups for the selected feature
        summary = (f"Highest median: **{medians.idxmax()}** ({medians.max():.2f}). "
                   f"Lowest median: **{medians.idxmin()}** ({medians.min():.2f}). ")
    # Else, if there is only one group in the filtered dataset, a message is displayed indicating that only one group matches the current filters. This informs users that they cannot compare multiple groups in this case.
    else:
        # Summary is a message indicating that only one group matches the current filters, and it informs users that they cannot compare multiple groups in this case. The caption also provides information about how to interpret the box plot, including the middle 50% of students and the median line.
        summary = "Only one group matches the current filters. "
    # The caption is added to the page to display the summary information about the highest and lowest median values among the groups, or the message indicating that only one group is available for comparison. The caption also provides information about how to interpret the box plot, including the middle 50% of students and the median line.
    st.caption(summary + "The box shows the middle 50% of students, the line inside it is the median.")

# With tab_dist: - creates a context for adding content to the "Distributions" tab. All Streamlit commands within this context will be applied to the "Distributions" tab, allowing for a clean and organized layout.
with tab_dist:
    # Section header: "How are values spread out?" - indicates that the following content relates to exploring the distribution of values for selected features in the dataset.
    section_header("How are values spread out?")
    # c1, c2 = st.columns(2) - creates two columns in the "Distributions" tab, each containing a different dropdown widget for selecting the feature to display in the histogram and the feature to split the histogram by. These dropdowns allow users to customize the histogram based on their interests and explore how values are distributed for selected features in the dataset.
    c1, c2 = st.columns(2)
    # Dist_col is a variable that stores the selected numeric feature for the X-axis of the histogram. The selectbox widget allows users to choose from the list of numeric columns, with "daily_screen_time_hours" set as the default selection. The friendly_name function is used to display user-friendly names for the features in the dropdown.
    dist_col = c1.selectbox(
        "Column", numeric_cols,
        index=numeric_cols.index("daily_screen_time_hours"),
        format_func=friendly_name,
    )
    # Split_col is a variable that stores the selected categorical feature for splitting the histogram into different groups. The selectbox widget allows users to choose from a predefined list of categorical features, with an additional option for "Nothing" (None) to indicate no splitting. The friendly_name function is used to display user-friendly names for the features in the dropdown.
    split_col = c2.selectbox(
        "Split by",
        [None, "device_free_before_bed", "has_part_time_job", "chronotype"],
        format_func=lambda c: "Nothing" if c is None else friendly_name(c),
    )
    # Bins is a variable that stores the number of bars (bins) to be used in the histogram. The slider widget allows users to select a value between 5 and 40, with a default value of 20. This provides users with control over the granularity of the histogram, allowing them to see more or fewer bars based on their preferences.
    # We need to use a slider instead of a selectbox because the number of bars is a continuous variable, and a slider allows users to easily select a value within a specified range. A selectbox would require users to choose from a predefined list of options, which may not be as flexible or user-friendly for this purpose.
    bins = st.slider("Number of bars", min_value=5, max_value=40, value=20)

    # Figure is a histogram figure created using Plotly Express. The px.histogram function is used to create the histogram based on the filtered dataset and the selected feature for the X-axis, as well as the selected feature for splitting the histogram into different groups. The number of bins is set based on the user's selection, and the title and labels for the axes are also set using user-friendly names.
    fig = px.histogram(
        filtered,
        x=dist_col,
        color=split_col,
        nbins=bins,
        barmode="overlay" if split_col else "relative",
        opacity=0.75 if split_col else 1.0,
        title=f"Distribution of {friendly_name(dist_col).lower()}",
        labels=LABELS,
    )
    # Update the layout of the histogram figure to set the Y-axis title to "Number of students" and adjust the gap between bars to 0.05. This improves the readability of the histogram and provides context for interpreting the data.
    fig.update_layout(yaxis_title="Number of students", bargap=0.05)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Median {friendly_name(dist_col).lower()}: {filtered[dist_col].median():.2f}. "
        f"{COLUMN_DESCRIPTIONS.get(dist_col, '')}."
    )

# With tab_corr: - creates a context for adding content to the "Correlations" tab. All Streamlit commands within this context will be applied to the "Correlations" tab, allowing for a clean and organized layout.
# The "Correlations" tab allows users to explore the relationships between different numeric features in the dataset by calculating and visualizing their correlation coefficients.
with tab_corr:
    # Section header: "Which columns move together?" - indicates that the following content relates to exploring the correlations between different numeric features in the dataset.
    section_header("Which columns move together?")
    # Default_cols is a list of numeric features that are pre-selected for the correlation analysis. These features are considered important for understanding the relationships between sleep, screen time, lifestyle habits, and CGPA. The TARGET variable (CGPA) is also included in this list to analyze its correlations with other features.
    default_cols = [
        "avg_sleep_hours_weekday", "sleep_quality_score", "daily_screen_time_hours",
        "caffeine_drinks_per_day", "weekly_study_hours", "class_attendance_percent", TARGET,
    ]
    # Corr_cols is a variable that stores the selected numeric features for the correlation analysis. The multiselect widget allows users to choose multiple features from the list of numeric columns, with the default_cols pre-selected. The friendly_name function is used to display user-friendly names for the features in the dropdown.
    corr_cols = st.multiselect(
        "Columns to include", numeric_cols, default=default_cols, format_func=friendly_name
    )
    
    # If the user selects fewer than two columns for the correlation analysis, an informational message is displayed to inform them that at least two columns are required to calculate and visualize correlations. This ensures that users understand the requirements for performing correlation analysis and encourages them to select additional features for meaningful results.
    if len(corr_cols) < 2:
        # st.info("Pick at least two columns to see their correlations.") - displays an informational message in the Streamlit app, prompting users to select at least two columns for the correlation analysis. This message helps guide users in making appropriate selections for meaningful correlation results.
        st.info("Pick at least two columns to see their correlations.")
    else:
        # Calculate the correlation matrix for the selected numeric features using the pandas corr() method. The resulting correlation coefficients are rounded to two decimal places for better readability. The correlation matrix shows how strongly each pair of features is related, with values ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation).
        corr = filtered[corr_cols].corr().round(2)
        # Names is a list of user-friendly names for the selected numeric features, generated using the friendly_name function. This list is used to label the axes of the correlation heatmap, making it easier for users to interpret the results.
        names = [friendly_name(c) for c in corr_cols]
        # Create a heatmap visualization of the correlation matrix using Plotly Express. The px.imshow function is used to create the heatmap, with the correlation coefficients displayed as text on the heatmap cells. The color scale is set to a diverging scale, where brown represents negative correlations, grey represents no correlation, and teal represents positive correlations. The title and aspect ratio of the heatmap are also set for better presentation.
        fig = px.imshow(
            corr,
            x=names,
            y=names,
            text_auto=True,  # print every value, so the colours aren't needed to read it
            zmin=-1,
            zmax=1,
            # Diverging scale: brown = negative, grey = none, teal = positive
            color_continuous_scale=["#BE7D59", "#4A4C4A", "#69A2B0"],
            title="Correlation between columns (-1 to +1)",
            aspect="auto",
        )
        # Streamlit command to display the correlation heatmap figure in the "Correlations" tab. The width parameter is set to "stretch" to make the chart responsive and fill the available horizontal space in the app layout.
        st.plotly_chart(fig, width="stretch")
        # A caption is added to the page to explain the interpretation of the correlation coefficients displayed in the heatmap. It informs users that values near +1 indicate that features move up together, values near -1 indicate that features move in opposite directions, and values near 0 indicate no straight-line relationship between the features.
        st.caption(
            "Values near +1 move up together, values near -1 move in opposite directions, "
            "and values near 0 have no straight-line relationship."
        )
