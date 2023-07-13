import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import curve_fit
import plotly.express as px
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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
    proj_df = pd.DataFrame(projection,columns=["Projection"])

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
    
    date_format = "%m/%d/%Y"
    start_date = datetime.strptime(all_data["Dates"].iloc[-1], date_format)
    proj_dates  = [start_date.strftime(date_format)] + [(start_date + relativedelta(months=1 * i)).strftime(date_format) for i in range(1, len(projection))]
    proj_df["Dates"] = proj_dates
    
    fig2 = px.line(proj_df, x = proj_df.index, y = proj_df["Projection"],log_y = scale,title = "Projected Curve")
    fig2.update_xaxes(title_text="Month")
    fig2.update_yaxes(title_text="Production")
    st.plotly_chart(fig2)
    return proj_df

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
    #st.write(time_fit, projection_time)
    #st.write(projection)

    proj_df = pd.DataFrame(projection,columns=["Projection"])
    
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
    
    
    date_format = "%m/%d/%Y"
    start_date = datetime.strptime(all_data["Dates"].iloc[-1], date_format)
    proj_dates  = [start_date.strftime(date_format)] + [(start_date + relativedelta(months=1 * i)).strftime(date_format) for i in range(1, len(projection))]
    proj_df["Dates"] = proj_dates
    
    
    fig2 = px.line(proj_df, x = proj_dates, y = proj_df["Projection"],log_y = scale,title = "Projected Curve")
    fig2.update_xaxes(title_text="Month")
    fig2.update_yaxes(title_text="Production")
    st.plotly_chart(fig2)

    return proj_df


st.write('**Upload a file with the columns "Date" and any of "Oil", "Gas", and "Water" ("Date" is assumed to be monthly)**')
uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Date" not in df.columns:
        st.error("Please upload a file with a date column.")

    with st.expander("Show uploaded data"):
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    tab1, tab2, tab3 = st.tabs(["Production","Cash Flow","Summary"])
    with tab1:
        c1, c2 = st.columns(2)
        
        with c1:
            starting_index = st.number_input("**Choose the index of the initial production:**",min_value = 0, max_value = len(df["Date"]),step=1)
            decline_type = st.radio("**Choose decline type:**",options=["Hyperbolic","Exponential"],index = 0,horizontal=True)

        with c2:
            projection_months = st.number_input("**Months to project**",min_value = 12, max_value = 120, value = 60)
            scale_metric = st.checkbox("**Use log scale**",value = True)
        st.divider()
        if decline_type == "Hyperbolic":
            projection = fit_hyperbolic_curve(df,start_index = starting_index,scale = scale_metric,p_months = projection_months)
        else:
            projection = fit_exponential_curve(df,start_index = starting_index,scale = scale_metric,p_months = projection_months)

    with tab2:
        co1, co2, co3 = st.columns(3)
        with co1:
            oil_price = st.number_input("Oil Price",min_value = 20, max_value = 200, value = 60,step = 1)
            st.number_input("Average Operating Expenses",min_value = 1000, max_value = 100000, value = 7000, step = 50)
        with co2:
            st.number_input("Gas/Oil Ratio (MCF/bbl)", max_value = 10.00,min_value = 0.00, value = 1.00, step = .05)
            st.number_input("Plugging Costs", min_value = 0, max_value = 200000, value = 60000, step = 1000)
        with co3:
            gas_price = st.number_input("Gas Price",min_value = 0.0, max_value = 10.0, value = 3.0,step = .1)

        st.slider("Choose months to use for cash flows", 1,len(projection), (1,len(projection)))

        st.divider()



    st.subheader("Projected Production")
    st.write(projection)
