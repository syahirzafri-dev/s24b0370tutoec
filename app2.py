import os
import random
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(page_title="GA Program Scheduling", layout="wide")
st.title("Genetic Algorithm Program Scheduling")
st.write("Configure Crossover Rate (`CO_R`) and Mutation Rate (`MUT_R`) for three trials to generate optimal program schedules.")

# 1. Load program_ratings.csv safely from the same folder as app2.py
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "program_ratings.csv")

if not os.path.exists(csv_path):
    st.error(f"Could not find 'program_ratings.csv' at: {csv_path}. Please ensure it is uploaded in the same GitHub folder as app2.py.")
    st.stop()

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Extract program names (first column) and time slots (remaining columns)
program_col = df.columns[0]
time_slots = list(df.columns[1:])
programs = df[program_col].astype(str).str.strip().tolist()

# Build rating lookup dictionary: ratings[program][time_slot] = float_value
ratings = {}
for _, row in df.iterrows():
    prog = str(row[program_col]).strip()
    ratings[prog] = {}
    for slot in time_slots:
        val = pd.to_numeric(row[slot], errors="coerce")
        ratings[prog][slot] = 0.0 if pd.isna(val) else float(val)

with st.expander("View Loaded Dataset (program_ratings.csv)", expanded=False):
    st.dataframe(df, use_container_width=True)


# 2. Genetic Algorithm Functions
def calculate_fitness(schedule):
    """Total rating across all scheduled time slots."""
    return sum(ratings[schedule[i]][time_slots[i]] for i in range(len(time_slots)))


def create_individual():
    """Create a random schedule (one program per time slot)."""
    return [random.choice(programs) for _ in range(len(time_slots))]


def tournament_selection(population, fitnesses, k=3):
    """Select the best individual among k random candidates."""
    selected_indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx][:]


def crossover(parent1, parent2, co_r):
    """Single-point crossover based on CO_R."""
    if random.random() < co_r and len(parent1) > 1:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1[:], parent2[:]


def mutate(individual, mut_r):
    """Randomly replace a program in each slot with probability MUT_R."""
    for i in range(len(individual)):
        if random.random() < mut_r:
            individual[i] = random.choice(programs)
    return individual


def run_genetic_algorithm(co_r, mut_r, pop_size=50, generations=100, seed=42):
    """Run the Genetic Algorithm and return the best schedule and its total rating."""
    random.seed(seed)
    population = [create_individual() for _ in range(pop_size)]

    best_schedule = None
    best_fitness = -1.0

    for _ in range(generations):
        fitnesses = [calculate_fitness(ind) for ind in population]

        # Track best solution (Elitism)
        for ind, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_schedule = ind[:]

        new_population = [best_schedule[:]]  # Keep elite individual

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2, co_r)
            new_population.append(mutate(child1, mut_r))
            if len(new_population) < pop_size:
                new_population.append(mutate(child2, mut_r))

        population = new_population

    return best_schedule, best_fitness


# 3. Sidebar Parameter Inputs (3 Trials)
st.sidebar.header("GA Parameter Settings")
st.sidebar.caption("CO_R range: 0.00 – 0.95 | MUT_R range: 0.01 – 0.05")

# Note: Default MUT_R is set to 0.02 so it stays within the required [0.01, 0.05] range without error
st.sidebar.subheader("Trial 1 (Default Baseline)")
co_r_1 = st.sidebar.slider("Trial 1 - CO_R", min_value=0.0, max_value=0.95, value=0.80, step=0.05)
mut_r_1 = st.sidebar.slider("Trial 1 - MUT_R", min_value=0.01, max_value=0.05, value=0.02, step=0.01)

st.sidebar.subheader("Trial 2")
co_r_2 = st.sidebar.slider("Trial 2 - CO_R", min_value=0.0, max_value=0.95, value=0.60, step=0.05)
mut_r_2 = st.sidebar.slider("Trial 2 - MUT_R", min_value=0.01, max_value=0.05, value=0.03, step=0.01)

st.sidebar.subheader("Trial 3")
co_r_3 = st.sidebar.slider("Trial 3 - CO_R", min_value=0.0, max_value=0.95, value=0.40, step=0.05)
mut_r_3 = st.sidebar.slider("Trial 3 - MUT_R", min_value=0.01, max_value=0.05, value=0.05, step=0.01)

with st.sidebar.expander("Advanced GA Settings"):
    pop_size = st.number_input("Population Size", min_value=10, max_value=500, value=50, step=10)
    generations = st.number_input("Generations", min_value=10, max_value=1000, value=100, step=10)

# 4. Run Trials and Display Schedule Tables
if st.button("Run Genetic Algorithm (All 3 Trials)", type="primary"):
    trials = [
        ("Trial 1", co_r_1, mut_r_1),
        ("Trial 2", co_r_2, mut_r_2),
        ("Trial 3", co_r_3, mut_r_3),
    ]

    cols = st.columns(3)
    summary_rows = []

    for idx, (trial_name, co_val, mut_val) in enumerate(trials):
        best_sched, best_score = run_genetic_algorithm(
            co_r=co_val,
            mut_r=mut_val,
            pop_size=int(pop_size),
            generations=int(generations),
            seed=42 + idx,
        )

        schedule_table = pd.DataFrame({
            "Time Slot": time_slots,
            "Scheduled Program": best_sched,
            "Rating": [ratings[best_sched[i]][time_slots[i]] for i in range(len(time_slots))],
        })

        summary_rows.append({
            "Trial": trial_name,
            "CO_R": co_val,
            "MUT_R": mut_val,
            "Total Rating (Fitness)": round(best_score, 2),
        })

        with cols[idx]:
            st.subheader(trial_name)
            st.write(f"**CO_R:** `{co_val}` | **MUT_R:** `{mut_val}`")
            st.metric("Total Schedule Rating", f"{best_score:.2f}")
            st.table(schedule_table)

    st.divider()
    st.subheader("Comparison Summary Across Trials")
    st.table(pd.DataFrame(summary_rows))
