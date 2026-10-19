"""
Predict page: enter a student's habits and get a predicted CGPA from the trained model.

Drafted with help from Claude (Anthropic).
"""
# Import time
from datetime import time

# Import pandas, plotly, and streamlit as variables
import pandas as pd
import plotly.express as px
import streamlit as st

# Import the custom components we have described
from components import page_header, section_header
from data_loading import COLUMN_DESCRIPTIONS, TARGET, friendly_name, load_clean_data
from model_loading import load_metrics, load_predictor

# CONSTANT
CARD_HEIGHT = 580

# Page header description is not a caption its a description.
page_header(
    "Predict",
    "Describe a student's sleep, screen and study habits, and the model estimates their CGPA.",
    isCaption=False
)

# Define dataframe, and anonymous variable to the returned structure of load_clean_data
df, _ = load_clean_data()
# Feature cols is a variable - using list comprehension - for the values 'c' in dataframe in columns IF c is not the target. We are using everything besides the CGPA
feature_cols = [c for c in df.columns if c != TARGET]  # the exact columns and order the model expects
# Unpack predictor, not_ready_reason with the returned structure of load_predictor
predictor, not_ready_reason = load_predictor()

# If predictor is None
if predictor is None:
    # Displat a streamlit warning using 'f strings'
    st.warning(
        f"**Live predictions aren't available yet.** {not_ready_reason} "
        "You can still try the form; the button turns on once the model is saved.",
        icon=":material/hourglass_empty:",
    )

# Input helpers: every widget gets its range and default from the real data,
# and its tooltip from the column descriptions

# Defining the number_slider function, the parameters col, step's default is float 0.1, integer is a bool where it is False
def number_slider(col: str, step: float = 0.1, integer: bool = False):
    # unpack the min, max, and median for all the column
    low, high, middle = df[col].min(), df[col].max(), df[col].median()
    # If it is an integer, then act accordingly 
    if integer:
        # Define low, high, middle, step = where we typecast into int, and define the step as 1.
        low, high, middle, step = int(low), int(high), int(round(middle)), 1
    else:
        # Otherwise, use float, and round the middle value to 1 place
        low, high, middle = float(low), float(high), round(float(middle), 1)
    # Return a slider with the friendly_name for the column, and the low, high, middle, step=step, and get the column description
    return st.slider(friendly_name(col), low, high, middle, step=step,
                     help=COLUMN_DESCRIPTIONS.get(col))

# Defining category_select(column a string)
def category_select(col: str):
    # Define options: Sort the dataframe [col] for all the unique 
    options = sorted(df[col].unique())
    # Most common value as sort the dataframe for the col, and get the mode and 0
    most_common = df[col].mode()[0]
    # Return the select box, with the friendly column, options, index = options.index(for the most common)
    return st.selectbox(friendly_name(col), options, index=options.index(most_common),
                        help=COLUMN_DESCRIPTIONS.get(col))

# Define the function yes_no with the parameter (col:str)
def yes_no(col: str):
    # Define most_common = df[col].mode()[0]
    most_common = df[col].mode()[0]
    # Return a radio button for each column 
    return st.radio(friendly_name(col), ["Yes", "No"], index=["Yes", "No"].index(most_common),
                    horizontal=True, help=COLUMN_DESCRIPTIONS.get(col))


# Layout: the form on the left, the result on the right

# Unpack form_col, result_col, the two streamlit columns and set the gap to be large
form_col, result_col = st.columns([2, 1], gap="large")

# With form column as context 
with form_col:
    # A form waits for the button before re-running, instead of re-running on every slider move
    with st.form("predict_form", border=False):
        # Values is a set 
        values = {}
        # Unpack left, right into two columns
        left, right = st.columns(2)

        # With the left container, ensure a border, and set a fixed height for all 
        with left.container(border=True, height=CARD_HEIGHT):
            st.markdown("**:material/person: About the student**")
            values["age"] = number_slider("age", integer=True)
            values["year_of_study"] = number_slider("year_of_study", integer=True)
            values["major_category"] = category_select("major_category")
            values["residence_type"] = category_select("residence_type")
            values["has_part_time_job"] = yes_no("has_part_time_job")
            values["semester_type"] = category_select("semester_type")
            

        # Same process
        with right.container(border=True, height=CARD_HEIGHT):
            st.markdown("**:material/bedtime: Sleep**")
            # bedtime is the time_input
            bedtime = st.time_input("Average bedtime", value=time(23, 0), step=900,
                                    help=COLUMN_DESCRIPTIONS["avg_bedtime_hour"])
            values["avg_bedtime_hour"] = bedtime.hour + bedtime.minute / 60  # 23:30 -> 23.5
            values["avg_sleep_hours_weekday"] = number_slider("avg_sleep_hours_weekday")
            values["weekend_sleep_delta_hours"] = number_slider("weekend_sleep_delta_hours")
            values["chronotype"] = category_select("chronotype")
            values["naps_per_week"] = number_slider("naps_per_week", integer=True)
            values["sleep_quality_score"] = number_slider("sleep_quality_score", integer=True)

        # Same process
        with left.container(border=True, height=CARD_HEIGHT):
            st.markdown("**:material/smartphone: Screens and lifestyle**")
            values["pre_sleep_screen_time_min"] = number_slider("pre_sleep_screen_time_min", integer=True)
            values["daily_screen_time_hours"] = number_slider("daily_screen_time_hours")
            values["primary_device"] = category_select("primary_device")
            values["device_free_before_bed"] = yes_no("device_free_before_bed")
            values["caffeine_drinks_per_day"] = number_slider("caffeine_drinks_per_day", integer=True)
            values["exercise_days_per_week"] = number_slider("exercise_days_per_week", integer=True)

        # Same process
        with right.container(border=True, height=CARD_HEIGHT):
            st.markdown("**:material/menu_book: Study and wellbeing**")
            values["energy_level"] = number_slider("energy_level", integer=True)
            values["concentration_difficulty"] = number_slider("concentration_difficulty", integer=True)
            values["weekly_study_hours"] = number_slider("weekly_study_hours")
            values["class_attendance_percent"] = number_slider("class_attendance_percent")

        # Submitted is a form submit button 
        submitted = st.form_submit_button(
            "Predict CGPA", type="primary", icon=":material/online_prediction:",
            disabled=predictor is None, width="stretch",
        )

# Display the result column
with result_col:
    # Section header is result 
    section_header("Result")

    # If not submitted
    if not submitted:
        # Display an informational message
        st.info("Fill in the form and press **Predict CGPA**.", icon=":material/arrow_back:")
        # Stop the
        st.stop()

    # One row with the same columns, in the same order, as the training data.
    # The Predictor applies the same scaling and one-hot encoding used in training.
    features = pd.DataFrame([values])[feature_cols]
    predicted = float(min(max(predictor.predict(features)[0], 0.0), 4.0))  # keep it on the 0-4 scale

    # Average is the target's mean
    average = df[TARGET].mean()
    # For the predicted cgpa contianer
    with st.container(border=True):
        # Streamlit metric 
        st.metric(
            "Predicted CGPA",
            f"{predicted:.2f} / 4.0",
            delta=f"{predicted - average:+.2f} vs. average student",
            icon=":material/school:",
        )
        # the mean of the target column less than the predicted
        share_below = (df[TARGET] < predicted).mean()
        # Caption message 
        st.caption(f"Higher than about **{share_below:.0%}** of students in the dataset.")

    # Show where the prediction falls among real students
    # Define fig as a histogram
    fig = px.histogram(
        df, x=TARGET, nbins=25, title="Where this prediction falls",
        labels={TARGET: "CGPA (0-4 scale)"},
    )
    # add a vline to fig
    fig.add_vline(x=predicted, line_width=3, line_dash="dash", line_color="#F1DAC4",
                  annotation_text="Prediction", annotation_position="top")
    # Update the layout for the yaxis
    fig.update_layout(yaxis_title="Number of students", bargap=0.05, height=320)
    # display the figure
    st.plotly_chart(fig, width="stretch")

    # Define and set the value of metrics to load_metrics
    metrics = load_metrics()
    # If metrics is not none
    if metrics is not None:
        # Define typical error as the minimum of metrics["mae"]
        typical_error = metrics["mae"].min()
        # Caption the error message using 'fstrings'
        st.caption(f"On unseen students, predictions are typically off by about "
                   f"±{typical_error:.2f} CGPA points.")
    # Add the caption for the page
    st.caption("This is a statistical estimate from a small dataset, not advice about any one student.")