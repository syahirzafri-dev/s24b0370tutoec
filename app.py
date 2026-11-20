import random
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="JIE42903 - Evolutionary Computing Assignment", layout="wide"
)

st.title("📺 TV Program Scheduling Using Genetic Algorithm")
st.markdown(
    "**Student Name:** MUHAMMAD ZAHIN BIN SALIMI | **Matric No:** S24B0389"
)
st.markdown(
    "**Course:** JIE42903 – Evolutionary Computing | **Lecturer:** Ts. Dr. Fakhitah"
    " Ridzuan"
)
st.divider()

# 1. Load and display modified dataset
st.subheader("📁 Program Ratings Dataset (Hour 6 – Hour 23)")
try:
  df_ratings = pd.read_csv("program_ratings.csv")
  st.dataframe(df_ratings, use_container_width=True)
except FileNotFoundError:
  st.error(
      "❌ 'program_ratings.csv' not found. Please upload it to your repository."
  )


# Genetic Algorithm Class tailored for Hourly Rating Matrix
class GeneticScheduler:

  def __init__(self, df, pop_size=50, generations=100):
    self.df = df
    self.pop_size = pop_size
    self.generations = generations
    self.hours = [col for col in df.columns if col != "Type of Program"]
    self.programs = df["Type of Program"].tolist()

    # Build lookup dictionary: (program, hour) -> rating
    self.ratings = {}
    for _, row in df.iterrows():
      prog = row["Type of Program"]
      for h in self.hours:
        self.ratings[(prog, h)] = float(row[h])

  def fitness(self, chromosome):
    # Sum the hourly ratings for the scheduled program at each hour
    return sum(
        self.ratings[(prog, self.hours[i])] for i, prog in enumerate(chromosome)
    )

  def run(self, co_r, mut_r):
    # Initialize population: each individual assigns a program to each hour slot
    population = [
        [random.choice(self.programs) for _ in range(len(self.hours))]
        for _ in range(self.pop_size)
    ]

    best_schedule = None
    best_fitness = -float("inf")

    for _ in range(self.generations):
      scored_pop = [(ind, self.fitness(ind)) for ind in population]
      scored_pop.sort(key=lambda x: x[1], reverse=True)

      if scored_pop[0][1] > best_fitness:
        best_fitness = scored_pop[0][1]
        best_schedule = scored_pop[0][0]

      # Selection (Top 50%)
      selected = [ind for ind, _ in scored_pop[: self.pop_size // 2]]
      next_generation = selected.copy()

      while len(next_generation) < self.pop_size:
        p1, p2 = random.sample(selected, 2)
        # Single-point Crossover
        if random.random() < co_r:
          cut = random.randint(1, len(self.hours) - 1)
          child = p1[:cut] + p2[cut:]
        else:
          child = p1.copy()

        # Mutation
        for i in range(len(child)):
          if random.random() < mut_r:
            child[i] = random.choice(self.programs)

        next_generation.append(child)

      population = next_generation

    # Format best schedule into a DataFrame table
    scheduled_rows = []
    for i, prog in enumerate(best_schedule):
      hour_label = self.hours[i]
      rating_val = self.ratings[(prog, hour_label)]
      scheduled_rows.append({
          "Time Slot": hour_label,
          "Scheduled Program": prog,
          "Hourly Rating": rating_val,
      })

    return pd.DataFrame(scheduled_rows), round(best_fitness, 2)


# Sidebar Parameter Controls
st.sidebar.header("⚙️ GA Parameters")
co_r = st.sidebar.slider(
    "CO_R (Crossover Rate)",
    min_value=0.0,
    max_value=0.95,
    value=0.8,
    step=0.05,
)
mut_r = st.sidebar.slider(
    "MUT_R (Mutation Rate)",
    min_value=0.01,
    max_value=0.05,
    value=0.02,
    step=0.01,
)

st.sidebar.divider()
run_trials_btn = st.sidebar.button("🚀 Run 3 Experimental Trials")

if "df_ratings" in locals():
  scheduler = GeneticScheduler(df_ratings)

  if run_trials_btn:
    st.subheader("📊 Experimental Trial Results")

    # Trial 1 (User Configuration)
    st.markdown(
        f"### Trial 1: CO_R = {co_r}, MUT_R = {mut_r} (User Configuration)"
    )
    df_1, fit_1 = scheduler.run(co_r, mut_r)
    st.success(f"Total Schedule Fitness Score: **{fit_1}**")
    st.dataframe(df_1, use_container_width=True)

    # Trial 2 (High Exploration)
    st.markdown("### Trial 2: CO_R = 0.90, MUT_R = 0.01 (High Exploration)")
    df_2, fit_2 = scheduler.run(0.90, 0.01)
    st.success(f"Total Schedule Fitness Score: **{fit_2}**")
    st.dataframe(df_2, use_container_width=True)

    # Trial 3 (High Diversity)
    st.markdown("### Trial 3: CO_R = 0.60, MUT_R = 0.05 (High Diversity)")
    df_3, fit_3 = scheduler.run(0.60, 0.05)
    st.success(f"Total Schedule Fitness Score: **{fit_3}**")
    st.dataframe(df_3, use_container_width=True)

    st.divider()
    st.subheader("📈 Summary Table")
    summary_df = pd.DataFrame({
        "Trial": ["Trial 1", "Trial 2", "Trial 3"],
        "CO_R": [co_r, 0.90, 0.60],
        "MUT_R": [mut_r, 0.01, 0.05],
        "Total Fitness Score": [fit_1, fit_2, fit_3],
    })
    st.dataframe(summary_df, use_container_width=True)
  else:
    st.info(
        "👆 Set your parameters in the sidebar and click **'Run 3 Experimental"
        " Trials'**."
    )
