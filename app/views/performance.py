"""
Model Performance page: metrics table, predicted vs. actual, residuals, and more.

Drafted with help from Claude (Anthropic).
"""
# Import sys and Path from pathlib to manipulate the system path for module imports.
# Import streamlit as st for building the web application and displaying interactive components.
import plotly.express as px
import streamlit as st
# From components import page_header and section_header for creating consistent page headers and section headers in the Streamlit app. These are custom built components that help maintain a uniform look and feel across different pages of the application.
from components import page_header, section_header
# From data_loading import friendly_name for converting column names into user-friendly names for display in the Streamlit app. This function helps improve the readability of the data presented to users.
from data_loading import friendly_name
# From model_loading import MODELS_DIR, load_feature_importance, load_loss_history, load_metrics, load_test_predictions for loading model performance metrics, test predictions, feature importance, and loss history from the specified models directory. These functions are used to retrieve the necessary data for displaying model performance information on the page.
from model_loading import (
    MODELS_DIR, load_feature_importance, load_loss_history, load_metrics, load_test_predictions,
)

# Establish a standard page header for the Model Performance page using the page_header function. This header includes the title "Model Performance" and a brief description of the page's purpose, which is to show how well the trained models predict CGPA on students they have never seen (the test set).
# It is not a caption, so the description is displayed in a larger font size and gray color, as per the design requirements.
page_header(
    "Model Performance",
    "How well the trained models predict CGPA on students they have never seen (the test set).",
    isCaption=False,
)

# Load the model performance metrics and test predictions using the load_metrics and load_test_predictions functions. These functions read the corresponding CSV files from the models directory and return pandas DataFrames containing the relevant data. The metrics DataFrame contains information about the performance of different models, while the predictions DataFrame contains the actual and predicted CGPA values for students in the test set.
metrics = load_metrics()
# Load the model performance metrics and test predictions using the load_metrics and load_test_predictions functions. These functions read the corresponding CSV files from the models directory and return pandas DataFrames containing the relevant data. The metrics DataFrame contains information about the performance of different models, while the predictions DataFrame contains the actual and predicted CGPA values for students in the test set.
predictions = load_test_predictions()

# Until the training code has saved its results, explain what's missing instead of crashing
if metrics is None or predictions is None:
    # Using streamlit info to display an informational message in the Streamlit app, indicating that no model results are available yet. The message informs users that once the training code saves its results, the page will automatically populate with the relevant data. It also specifies that the expected files are `metrics.csv` and `test_predictions.csv` located in the models directory, and refers users to the top of `app/model_loading.py` for the exact format of these files.
    st.info(
        "**No model results yet.** Once the training code saves its results, this page fills in "
        f"automatically. It expects `metrics.csv` and `test_predictions.csv` in `{MODELS_DIR.name}/`. "
        "See the top of `app/model_loading.py` for the exact format.",
        icon=":material/hourglass_empty:",
    )
    # Stop the execution of the Streamlit app to prevent further processing and rendering of the page. This ensures that users are not presented with incomplete or missing data, and they are informed about the current state of the model results.
    st.stop()

# The best model is the one with the lowest RMSE (smallest typical error)
best = metrics.loc[metrics["rmse"].idxmin()]

# Best model at a glance
# Use st.markdown to display the best model's name and configuration in bold format. This provides users with a quick overview of the top-performing model based on the lowest RMSE value, allowing them to easily identify which model is currently considered the best for predicting CGPA.
st.markdown(f"Best model: **{best['model']}** ({best['configuration']})")
# Establish a card style dictionary to define the appearance of the metric cards displayed on the page. The dictionary specifies that the cards should have a border and a fixed height of 130 pixels. This consistent styling helps maintain a uniform look for the metric cards, making it easier for users to read and compare the displayed metrics.
CARD_STYLE = {"border": True, "height": 130}
# Unpack the four columns for displaying the model performance metrics (MAE, RMSE, R², and Training time) using st.columns. Each column will contain a metric card that presents the corresponding value for the best model, allowing users to quickly assess the model's performance at a glance.
c1, c2, c3, c4 = st.columns(4)
# c1 metric displays the Mean Absolute Error (MAE) of the best model, formatted to three decimal places. The metric card includes an icon representing a straight line and a tooltip explaining that MAE indicates the average deviation of predictions from actual CGPA values. The CARD_STYLE dictionary is applied to ensure consistent styling for the metric card.
c1.metric("MAE", f"{best['mae']:.3f}", icon=":material/straighten:",
          help="Mean absolute error: on average, predictions are off by this many CGPA points.",
          **CARD_STYLE)
# c2 metric displays the Root Mean Squared Error (RMSE) of the best model, formatted to three decimal places. The metric card includes an icon representing mathematical functions and a tooltip explaining that RMSE is similar to MAE but penalizes larger prediction errors more heavily. The CARD_STYLE dictionary is applied to ensure consistent styling for the metric card.
c2.metric("RMSE", f"{best['rmse']:.3f}", icon=":material/functions:",
          help="Root mean squared error: like MAE, but punishes large mistakes more.",
          **CARD_STYLE)
# c3 metric displays the R² value of the best model, formatted to two decimal places. The metric card includes an icon representing a percentage and a tooltip explaining that R² indicates the share of variation in CGPA that the model explains, with 1.0 being perfect and 0 indicating no better than guessing the average. The CARD_STYLE dictionary is applied to ensure consistent styling for the metric card.
c3.metric("R²", f"{best['r2']:.2f}", icon=":material/percent:",
          help="Share of the variation in CGPA the model explains (1.0 = perfect, 0 = no better "
               "than always guessing the average).",
          **CARD_STYLE)
# c4 metric displays the training time of the best model, formatted to two decimal places in seconds. The metric card includes an icon representing a timer and applies the CARD_STYLE dictionary for consistent styling. This metric provides users with information about how long it took to train the best-performing model, which can be useful for understanding the computational resources required for model training.
c4.metric("Training time", f"{best['train_time_s']:.2f} s", icon=":material/timer:",
          **CARD_STYLE)

# Write an empty line to create some vertical space between the metric cards and the subsequent content on the page. This improves the visual layout and readability of the page, making it easier for users to focus on the different sections of information presented.
st.write("")

# Unpack the four tabs for displaying different aspects of model performance using st.tabs. Each tab will contain specific content related to model comparison, predicted vs. actual CGPA, residuals, and additional insights into features and training. This tabbed layout allows users to easily navigate between different sections of information without overwhelming them with too much content at once.
tab_compare, tab_fit, tab_residuals, tab_extra = st.tabs(
    [
        ":material/leaderboard: Model comparison",
        ":material/target: Predicted vs. actual",
        ":material/ssid_chart: Residuals",
        ":material/insights: Features & training",
    ]
)

# Create a context for adding content to the "Model comparison" tab using the with statement. All Streamlit commands within this context will be applied to the "Model comparison" tab, allowing for a clean and organized layout of the model performance metrics and visualizations.
with tab_compare:
    # Section header for the "Model comparison" tab, indicating that the following content relates to comparing the performance of different models on the test set. This header helps users understand the purpose of the tab and what information they can expect to find within it.
    section_header("All models on the test set")
    # rename the metrics columns to be more user-friendly for display in the Streamlit app. The renamed columns include "Model", "Configuration", "MAE", "RMSE", "R²", and "Training time (s)". This improves the readability of the metrics table, making it easier for users to understand the performance metrics of different models at a glance.
    table = metrics.rename(columns={
        "model": "Model", "configuration": "Configuration", "mae": "MAE",
        "rmse": "RMSE", "r2": "R²", "train_time_s": "Training time (s)",
    })
    # Display the metrics table in the Streamlit app using st.dataframe. The table is rounded to three decimal places for better readability, and the index is hidden to focus on the relevant performance metrics. The width parameter is set to "stretch" to make the table responsive and fill the available horizontal space in the app layout.
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    # Caption for the metrics table, explaining that lower MAE and RMSE values indicate better model performance, while higher R² values indicate better model performance. It also clarifies that the test set was used once for this final comparison, providing context for interpreting the displayed metrics.
    st.caption("Lower MAE and RMSE are better. Higher R² is better. The test set was used once, "
               "for this final comparison.")

    # Metric choice radio button for selecting the metric to compare models by (MAE, RMSE, or R²). The radio button is displayed horizontally, and a format function is used to display user-friendly names for the metrics. This allows users to easily switch between different performance metrics for model comparison.
    metric_choice = st.radio(
        "Compare models by", ["mae", "rmse", "r2"], horizontal=True,
        format_func=lambda m: {"mae": "MAE", "rmse": "RMSE", "r2": "R²"}[m],
    )
    # Figure is a bar chart created using Plotly Express to visualize the selected performance metric (MAE, RMSE, or R²) for each model. The x-axis represents the model names, the y-axis represents the selected metric values, and the bars are annotated with the metric values formatted to three decimal places. The chart title and axis labels are set based on the selected metric, providing users with a clear visual comparison of model performance.
    fig = px.bar(
        metrics, x="model", y=metric_choice, text_auto=".3f",
        title=f"{metric_choice.upper().replace('R2', 'R²')} by model",
        labels={"model": "Model", metric_choice: metric_choice.upper().replace("R2", "R²")},
    )
    # Update the layout of the bar chart figure to set the Y-axis title based on the selected metric, adjust the gap between bars to 0.05, and set the height of the chart to 400 pixels. This improves the readability of the chart and provides context for interpreting the data.
    st.plotly_chart(fig, width="stretch")

# Tabs for predicted vs. actual, residuals, and extra info
with tab_fit:
    # Section header for the "Predicted vs. actual" tab, indicating that the following content relates to comparing the predicted CGPA values from the best model against the actual CGPA values in the test set. This header helps users understand the purpose of the tab and what information they can expect to find within it.
    section_header("Predicted vs. actual CGPA")
    # Scatter plot created using Plotly Express to visualize the relationship between actual and predicted CGPA values for students in the test set. The x-axis represents the actual CGPA values, the y-axis represents the predicted CGPA values, and the points are displayed with an opacity of 0.8. The chart title includes the best model's name, and axis labels are set to indicate actual and predicted CGPA values. A dashed diagonal line is added to represent perfect predictions, allowing users to easily assess how well the model's predictions align with the actual values.
    fig = px.scatter(
        predictions, x="actual", y="predicted", opacity=0.8,
        title=f"Predicted vs. actual CGPA ({best['model']}, test set)",
        labels={"actual": "Actual CGPA", "predicted": "Predicted CGPA"},
    )
    # Update traces is a function that performs an update operation on all traces that specify criteria (or all if none are provided)
    fig.update_traces(marker={"size": 8})
    # The diagonal: points on this line are perfect predictions
    # Low is the minimum of all predictions['actual'] and  predicitions['predicted']
    low = min(predictions["actual"].min(), predictions["predicted"].min())
    # High is the maximum of all predicitions['acutal'] and predicitions ['predicted']
    high = max(predictions["actual"].max(), predictions["predicted"].max())
    # We are adding a shape to figure, a line figure that begins from the low (x0, y0) and the second points are the high (x1, y1)
    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high,
                  line={"dash": "dash", "width": 2, "color": "#F1DAC4"})
    # Plotting the chart
    st.plotly_chart(fig, width="stretch")
    # A caption for the char explaining everything properly
    st.caption("Each dot is one student in the test set. Dots on the dashed line were predicted "
               "perfectly; the further a dot is from the line, the bigger the error.")

# With tab residuals is a tab  where we can see any residuals/errors
with tab_residuals:
    # Section header to display information there
    section_header("Prediction errors (residuals)")
    # Residuals is a variables where it assigns new coloumns to a datafram, in thise case, the difference between the actual values and the predicted values
    residuals = predictions.assign(residual=predictions["actual"] - predictions["predicted"])
    # Initialise and unpack two variables, left and right
    left, right = st.columns(2, gap="large")

    # For left, we will use a scatter plot to plot the values of the residual and the predicted CGPA 
    with left:
        # Define figure as px.scatter plot
        fig = px.scatter(
            residuals, x="predicted", y="residual", opacity=0.8,
            title="Residuals vs. predicted CGPA",
            labels={"predicted": "Predicted CGPA", "residual": "Residual (actual - predicted)"},
        )
        # Update all traces
        fig.update_traces(marker={"size": 8})
        # Add a horizontal line, dashed, at the y-coordinate
        fig.add_hline(y=0, line_dash="dash", line_color="#F1DAC4")
        # Display an interactive chart with fig as the chart
        st.plotly_chart(fig, width="stretch")

    # For right, we will be using a histogram to depict the distribution of residuals
    with right:
        # Define fig as px.histogram
        fig = px.histogram(
            residuals, x="residual", nbins=20, title="Distribution of residuals",
            labels={"residual": "Residual (actual - predicted)"},
        )
        # Update the layout of fig, with the title of y axis being "Number of students"; bargap shows the space between adjacent bins
        fig.update_layout(yaxis_title="Number of students", bargap=0.05)
        st.plotly_chart(fig, width="stretch")
    # Captioning the graph and using 'f strings' to format it to 3 decimal points 
    st.caption(
        "A good model has residuals scattered randomly around 0 with no pattern. "
        f"Here, the average residual is {residuals['residual'].mean():+.3f} CGPA points."
    )

# For the tab extra segment
with tab_extra:
    # Define importance as the return value (either pandas dataframe or none)
    importance = load_feature_importance()
    # Define history as the return value (either pandas dataframe or none)
    history = load_loss_history()

    # If both values are None
    if importance is None and history is None:
        # Return an informational message detailing the reason
        st.info("Feature importances and training curves will show here once the training "
                "code saves `feature_importance.csv` or `loss_history.csv`.")

    # If importance is not None, display the accurate data
    if importance is not None:
        # Add a section header
        section_header("Which features matter most?")
        # n_top represents the value of a slider, with 5 being the minimum, the length of the importance dataframe being the max, and the step being minimum of 10, or the length of importance (to ensure bounds)
        n_top = st.slider("Features to show", 5, len(importance), min(10, len(importance)))
        # Sort by size (ignoring the sign), keep the biggest; the largest ends up at the top
        top = importance.reindex(importance["importance"].abs().sort_values().index).tail(n_top)
        # Assign a name to columns and map each value to a friendly name
        top = top.assign(name=top["feature"].map(friendly_name))
        # Define fig as a bar chart
        fig = px.bar(
            top, x="importance", y="name", orientation="h",
            title=f"Top {n_top} features ({best['model']})",
            labels={"importance": "Importance", "name": "Feature"},
        )
        # Display the chart
        st.plotly_chart(fig, width="stretch")
    # If history is not None
    if history is not None:
        # Display the information regarding History
        section_header("Training and validation loss")
        # Long is a variable where we are transitioning the current history dataframe from a wide into a long form. This is best used when one or more columns are identifier variables, while other columns are considered measured variables.
        # The other columns are unpivoted and switched into rows
        long = history.melt(id_vars="epoch", value_vars=["loss", "val_loss"],
                            var_name="set", value_name="value")
        # Access the long [set] column and map the values from loss to training, and val loss to validation
        long["set"] = long["set"].map({"loss": "Training", "val_loss": "Validation"})
        # Define fig as a line graph. 
        fig = px.line(
            long, x="epoch", y="value", color="set", symbol="set",
            title="Loss per epoch", labels={"epoch": "Epoch", "value": "Loss (MSE)", "set": "Data"},
        )
        # Chart everything using streamlit
        st.plotly_chart(fig, width="stretch")
        # Here we are captioning the chart
        st.caption("If validation loss rises while training loss keeps falling, the network is "
                   "overfitting.")
