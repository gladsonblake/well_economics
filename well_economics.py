import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import curve_fit
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title = 'Curve Fitting',layout="wide")

def fit_hyperbolic_curve(df,start_index,scale, p_months):
    
    def hyperbolic_decline(t, b, di):
        return qi / ((1 + b * di * t) ** (1 / b))

    max_index = start_index
    qi = df["Oil"][max_index]
    # Define your data points (time and production rate)
    dates = df["Date"][max_index:].reset_index(drop=True)
    production = df["Oil"][max_index:].reset_index(drop=True)

    # Convert the dates to numerical values
    time = [(datetime.strptime(date, "%m/%d/%Y") - datetime.strptime(dates[0], "%m/%d/%Y")).days for date in dates]

    # Convert the data to numpy arrays
    time = np.array(time)
    production = np.array(production)

    bounds = ([0, 0], [2, 20])
    b_init = 1.0
    di_init = 0.02

    # Perform curve fitting
    params, _ = curve_fit(hyperbolic_decline, time, production,p0 = [b_init, di_init],bounds = bounds,method = "trf")

    # Extract the fitted parameters
    b_fit, di_fit = params

    # Generate a time array for the fitted curve
    time_fit = np.linspace(min(time), max(time), len(dates))

    # Calculate the production rate using the fitted parameters
    production_fit = hyperbolic_decline(time_fit, b_fit, di_fit)
    
    projection_time = np.linspace(max(time_fit) ,int(round(max(time)+p_months*30.437,0)),p_months+1)

    projection = hyperbolic_decline(projection_time, b_fit, di_fit)
    
    #st.write(projection)
    proj_df = pd.DataFrame(projection)
    

    st.write(f"Initial Production = {qi.round(2)}, b_0 = {b_fit.round(2)}, Initial decline = {di_fit.round(4)*100}%")

    all_data = {
        'Dates': dates,
        'Production': production,
        'Fitted Production': production_fit[:len(dates)]
    }
    mse = mean_squared_error(all_data["Production"], all_data["Fitted Production"])
    rmse = np.sqrt(mse)
    st.write(f"RMSE = {rmse}")
    # Create the figure object and add the traces to it
    fig = px.line(all_data, x = all_data["Dates"], y = ["Production","Fitted Production"],log_y = scale,title = "Fitted Curve")
    st.plotly_chart(fig)
    
    fig2 = px.line(proj_df, x = proj_df.index, y = proj_df[0],log_y = scale,title = "Projected Curve")
    fig2.update_xaxes(title_text="Month")
    fig2.update_yaxes(title_text="Production")
    st.plotly_chart(fig2)
    return projection

def fit_exponential_curve(df,start_index,scale, p_months):
    """
    Displays fitted curve ad projection graph
    
    Returns: an array with the projected production for the specified number of months
    """
    def exponential_decline(t, d):
        return qi * (1 - d)**t
    
    max_index = start_index
    qi = df["Oil"][max_index]
    # Define your data points (time and production rate)
    dates = df["Date"][max_index:].reset_index(drop=True)
    production = df["Oil"][max_index:].reset_index(drop=True)

    # Convert the dates to numerical values
    time = [(datetime.strptime(date, "%m/%d/%Y") - datetime.strptime(dates[0], "%m/%d/%Y")).days for date in dates]

    # Convert the data to numpy arrays
    time = np.array(time)
    production = np.array(production)
    
    bounds = ([0,.75])
    d_init = 0.05

    # Perform curve fitting
    params, _ = curve_fit(exponential_decline, time, production,p0 = d_init,bounds = bounds,method = "trf")

    # Extract the fitted parameters
    d_fit = params[0]

    # Generate a time array for the fitted curve
    time_fit = np.linspace(min(time), max(time), len(dates))
    # Calculate the production rate using the fitted parameters
    production_fit = exponential_decline(time_fit, d_fit)

    
    projection_time = np.linspace(max(time_fit) ,int(round(max(time)+p_months*30.437,0)),p_months+1)

    projection = exponential_decline(projection_time, d_fit)
    st.write(time_fit, projection_time)
    #st.write(projection)
    proj_df = pd.DataFrame(projection)

    st.write(f"Initial Production = {qi.round(2)}, Terminal decline = {d_fit.round(4)*100}%")
    
    all_data = {
        'Dates': dates,
        'Production': production,
        'Fitted Production': production_fit[:len(dates)]
    }
    mse = mean_squared_error(all_data["Production"], all_data["Fitted Production"])
    rmse = np.sqrt(mse)
    st.write(f"RMSE = {rmse}")
    
    
    fig = px.line(all_data, x = all_data["Dates"], y = ["Production","Fitted Production"],log_y = scale,title = "Fitted Curve")
    st.plotly_chart(fig)
    
    fig2 = px.line(proj_df, x = proj_df.index, y = proj_df[0],log_y = scale,title = "Projected Curve")
    fig2.update_xaxes(title_text="Month")
    fig2.update_yaxes(title_text="Production")
    st.plotly_chart(fig2)
    return projection


st.write('**Upload a file with the columns "Date" and any of "Oil", "Gas", and "Water" ("Date" is assumed to be monthly)**')
uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Date" not in df.columns:
        st.error("Please upload a file with a date column.")

    decline_type = st.radio("**Choose decline type:**",options=["Hyperbolic","Exponential"],index = 0,horizontal=True)
    starting_index = st.number_input("**Choose the index of the initial production:**",min_value = 0, max_value = len(df["Date"]),step=1)
    scale_metric = st.checkbox("**Use log scale**",value = True)
    projection_months = st.number_input("**Months to project**",min_value = 12, max_value = 120, value = 60)
    if decline_type == "Hyperbolic":
        projection = fit_hyperbolic_curve(df,start_index = starting_index,scale = scale_metric,p_months = projection_months)
    else:
        projection = fit_exponential_curve(df,start_index = starting_index,scale = scale_metric,p_months = projection_months)

st.subheader("Projected_Production")
st.write(projection)
