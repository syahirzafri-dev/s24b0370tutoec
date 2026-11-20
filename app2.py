import random
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cinema Showtime Scheduler (GA)",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Cinema Multiplex Showtime Scheduler using Genetic Algorithm")
st.markdown(
    "Optimize daily cinema hall schedules using an Evolutionary Genetic Algorithm "
    "to maximize audience occupancy ratings while maintaining genre variety."
)

# ---------------------------------------------------------
# 2. LOAD & PARSE CSV DATA
# ---------------------------------------------------------
@st.cache_data
def load_ratings(file_path="program_ratings.csv"):
    df = pd.read_csv(file_path)
    # First column = Movie/Program names; Remaining columns = Time Slots
    programs = df.iloc[:, 0].astype(str).tolist()
    time_slots = list(df.columns[1:])
    ratings_matrix = df.iloc[:, 1:].astype(float).values
    return df, programs, time_slots, ratings_matrix

try:
    raw_df, programs, time_slots, ratings_matrix = load_ratings("program_ratings.csv")
except FileNotFoundError:
    st.error("⚠️ `program_ratings.csv` not found. Please ensure it is in the same directory as `app.py`.")
    st.stop()

num_programs = len(programs)
num_slots = len(time_slots)

# ---------------------------------------------------------
# 3. GENETIC ALGORITHM FUNCTIONS
# ---------------------------------------------------------
def calculate_fitness(chromosome, ratings_matrix, repeat_penalty=0.25):
    """
    Fitness = Sum of audience ratings across all time slots
              minus a penalty if the exact same movie/genre plays consecutively.
    """
    total_rating = 0.0
    for slot_idx, prog_idx in enumerate(chromosome):
        total_rating += ratings_matrix[prog_idx][slot_idx]
        # Penalize consecutive identical genres to encourage schedule variety
        if slot_idx > 0 and chromosome[slot_idx] == chromosome[slot_idx - 1]:
            total_rating -= repeat_penalty
    return round(total_rating, 4)


def initialize_population(pop_size, num_slots, num_programs):
    return [
        [random.randint(0, num_programs - 1) for _ in range(num_slots)]
        for _ in range(pop_size)
    ]


def tournament_selection(population, fitnesses, k=3):
    selected_indices = random.sample(range(len(population)), k)
    best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx][:]


def crossover(parent1, parent2, co_r):
    if random.random() < co_r and len(parent1) > 1:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1[:], parent2[:]


def mutate(chromosome, mut_r, num_programs):
    for i in range(len(chromosome)):
        if random.random() < mut_r:
            chromosome[i] = random.randint(0, num_programs - 1)
    return chromosome


def run_genetic_algorithm(
    ratings_matrix,
    co_r=0.8,
    mut_r=0.2,
    pop_size=100,
    generations=200,
    repeat_penalty=0.25,
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_progs, num_slts = ratings_matrix.shape
    population = initialize_population(pop_size, num_slts, num_progs)

    best_chromosome = None
    best_fitness = -float("inf")
    fitness_history = []

    for _ in range(generations):
        fitnesses = [calculate_fitness(ind, ratings_matrix, repeat_penalty) for ind in population]

        # Track best solution
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_chromosome = population[gen_best_idx][:]

        fitness_history.append(best_fitness)

        # Elitism: carry forward top 2 individuals
        sorted_indices = np.argsort(fitnesses)[::-1]
        next_population = [
            population[sorted_indices[0]][:],
            population[sorted_indices[1]][:]
        ]

        # Generate offspring
        while len(next_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2, co_r)
            c1 = mutate(c1, mut_r, num_progs)
            c2 = mutate(c2, mut_r, num_progs)
            next_population.append(c1)
            if len(next_population) < pop_size:
                next_population.append(c2)

        population = next_population

    return best_chromosome, best_fitness, fitness_history


# ---------------------------------------------------------
# 4. STREAMLIT SIDEBAR CONTROLS (PARAMETERS)
# ---------------------------------------------------------
st.sidebar.header("⚙️ GA Parameter Settings")

co_r = st.sidebar.slider(
    "Crossover Rate (CO_R)",
    min_value=0.00,
    max_value=0.95,
    value=0.80,
    step=0.05,
    help="Probability of combining two parent schedules (Default: 0.80, Max: 0.95)."
)

mut_r = st.sidebar.slider(
    "Mutation Rate (MUT_R)",
    min_value=0.01,
    max_value=0.50,
    value=0.20,
    step=0.01,
    help="Probability of randomly mutating a time slot (Default: 0.20; test 0.01–0.05 for low-mutation trials)."
)

with st.sidebar.expander("Advanced Algorithm Settings"):
    pop_size = st.number_input("Population Size", min_value=20, max_value=500, value=100, step=10)
    generations = st.number_input("Number of Generations", min_value=50, max_value=1000, value=200, step=50)
    repeat_penalty = st.number_input("Consecutive Repeat Penalty", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    use_seed = st.checkbox("Fix Random Seed (Reproducible Trials)", value=False)
    seed_val = st.number_input("Seed Value", value=42, step=1) if use_seed else None

run_btn = st.sidebar.button("🚀 Run Genetic Algorithm", type="primary", use_container_width=True)

# Initialize session state for storing the 3 experimental trials
if "trials" not in st.session_state:
    st.session_state.trials = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# ---------------------------------------------------------
# 5. EXECUTE & DISPLAY RESULTS
# ---------------------------------------------------------
with st.expander("📊 View Modified Audience Rating Dataset (`program_ratings.csv`)"):
    st.dataframe(raw_df, use_container_width=True)

if run_btn or st.session_state.latest_result is None:
    best_sched, best_fit, history = run_genetic_algorithm(
        ratings_matrix=ratings_matrix,
        co_r=co_r,
        mut_r=mut_r,
        pop_size=pop_size,
        generations=generations,
        repeat_penalty=repeat_penalty,
        seed=seed_val
    )

    schedule_rows = []
    raw_sum = 0.0
    for slot_idx, prog_idx in enumerate(best_sched):
        rating_val = ratings_matrix[prog_idx][slot_idx]
        raw_sum += rating_val
        is_repeat = (slot_idx > 0 and best_sched[slot_idx] == best_sched[slot_idx - 1])
        schedule_rows.append({
            "Time Slot": time_slots[slot_idx],
            "Scheduled Movie / Genre": programs[prog_idx],
            "Audience Rating": round(rating_val, 2),
            "Consecutive Repeat?": "Yes (-Penalty)" if is_repeat else "No"
        })

    st.session_state.latest_result = {
        "co_r": co_r,
        "mut_r": mut_r,
        "best_fitness": best_fit,
        "raw_rating_sum": round(raw_sum, 4),
        "schedule_df": pd.DataFrame(schedule_rows),
        "history
