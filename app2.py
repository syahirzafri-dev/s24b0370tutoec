import streamlit as st
import pandas as pd
import random

st.title("Genetic Algorithm Program Scheduling")

# 1. Load the modified CSV file
df = pd.read_csv("modified_ratings.csv")

# 2. Sidebar / Main Parameter Inputs
st.sidebar.header("GA Parameters")
CO_R = st.sidebar.slider(
    "Crossover Rate (CO_R)", 
    min_value=0.0, 
    max_value=0.95, 
    value=0.80, 
    step=0.05
)

MUT_R = st.sidebar.slider(
    "Mutation Rate (MUT_R)", 
    min_value=0.01, 
    max_value=0.50,  # Adjusted from 0.05 so default 0.20 does not trigger a Streamlit error
    value=0.20, 
    step=0.01
)

# 3. Run GA when button is clicked
if st.button("Generate Schedule"):
    # --- PASTE YOUR WEEK 4 GA FUNCTIONS HERE ---
    # Pass CO_R, MUT_R, and df into your GA function
    # best_schedule, best_fitness = run_genetic_algorithm(df, CO_R, MUT_R)
    
    # Example placeholder DataFrame for display:
    schedule_df = pd.DataFrame({
        "Time Slot": ["08:00", "09:00", "10:00"],
        "Scheduled Program": ["Program A", "Program B", "Program C"],
        "Rating": [8.5, 9.1, 7.8]
    })

    st.subheader(f"Best Schedule (CO_R = {CO_R}, MUT_R = {MUT_R})")
    # st.metric("Total Fitness / Rating", f"{best_fitness:.2f}")
    st.table(schedule_df)  # Or st.dataframe(schedule_df, use_container_width=True)
