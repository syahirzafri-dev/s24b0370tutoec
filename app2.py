import os
import random
import pandas as pd
import streamlit as st

# 1. Page Configuration & Custom Styling
st.set_page_config(
    page_title="PrimeGrid  | Broadcast Schedule Optimizer",
    page_icon="📺",
    layout="wide",
)

st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
                padding: 20px 25px; border-radius: 12px; color: white; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">📺 PrimeGrid </h1>
        <p style="margin: 5px 0 0 0; font-size: 1.05rem; opacity: 0.9;">
            Evolutionary Broadcast & Sports Lineup Optimizer • Powered by Genetic Algorithms
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Load program_ratings.csv safely
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "program_ratings.csv")

if not os.path.exists(csv_path):
    st.error(f"Missing 'program_ratings.csv' in {os.path.dirname(csv_path)}. Please upload it to your GitHub repo.")
    st.stop()

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

program_col = df.columns[0]
time_slots = list(df.columns[1:])
programs = df[program_col].astype(str).str.strip().tolist()

ratings = {}
for _, row in df.iterrows():
    prog = str(row[program_col]).strip()
    ratings[prog] = {}
    for slot in time_slots:
        val = pd.to_numeric(row[slot], errors="coerce")
        ratings[prog][slot] = 0.0 if pd.isna(val) else float(val)


def format_program_label(prog_name):
    """Add intuitive broadcast icons based on program type."""
    name_lower = prog_name.lower()
    if "soccer" in name_lower or "sport" in name_lower:
        return f"⚽ {prog_name}"
    if "boxing" in name_lower:
        return f"🥊 {prog_name}"
    if "news" in name_lower:
        return f"📰 {prog_name}"
    if "movie" in name_lower:
        return f"🎬 {prog_name}"
    if "music" in name_lower:
        return f"🎵 {prog_name}"
    if "doc" in name_lower:
        return f"🌍 {prog_name}"
    if "reality" in name_lower or "series" in name_lower:
        return f"🎭 {prog_name}"
    return f"📺 {prog_name}"


def get_slot_status(rating_val):
    """Classify slot performance."""
    if rating_val >= 0.5:
        return "🔥 Prime Peak"
    if rating_val >= 0.3:
        return "⭐ High"
    return "🌙 Standard"


# 3. Genetic Algorithm Engine
def calculate_fitness(schedule):
    return sum(ratings[schedule[i]][time_slots[i]] for i in range(len(time_slots)))


def create_individual():
    return [random.choice(programs) for _ in range(len(time_slots))]


def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(selected, key=lambda idx: fitnesses[idx])
    return population[best_idx][:]


def crossover(parent1, parent2, co_r):
    if random.random() < co_r and len(parent1) > 1:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1[:], parent2[:]


def mutate(individual, mut_r):
    for i in range(len(individual)):
        if random.random() < mut_r:
            individual[i] = random.choice(programs)
    return individual


def run_genetic_algorithm(co_r, mut_r, pop_size=50, generations=100, seed=42):
    random.seed(seed)
    population = [create_individual() for _ in range(pop_size)]
    best_schedule = None
    best_fitness = -1.0

    for _ in range(generations):
        fitnesses = [calculate_fitness(ind) for ind in population]
        for ind, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_schedule = ind[:]

        new_population = [best_schedule[:]]
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2, co_r)
            new_population.append(mutate(c1, mut_r))
            if len(new_population) < pop_size:
                new_population.append(mutate(c2, mut_r))
        population = new_population

    return best_schedule, best_fitness


# 4. Sidebar Controls
st.sidebar.header("🎛️ Optimizer Controls")
st.sidebar.caption("Tune Crossover (`CO_R`: 0.00–0.95) & Mutation (`MUT_R`: 0.01–0.05)")

st.sidebar.subheader("Trial 1 • High Exploitation")
co_r_1 = st.sidebar.slider("Trial 1 - CO_R", 0.0, 0.95, 0.80, 0.05)
mut_r_1 = st.sidebar.slider("Trial 1 - MUT_R", 0.01, 0.05, 0.02, 0.01)

st.sidebar.subheader("Trial 2 • Balanced Search")
co_r_2 = st.sidebar.slider("Trial 2 - CO_R", 0.0, 0.95, 0.60, 0.05)
mut_r_2 = st.sidebar.slider("Trial 2 - MUT_R", 0.01, 0.05, 0.03, 0.01)

st.sidebar.subheader("Trial 3 • High Exploration")
co_r_3 = st.sidebar.slider("Trial 3 - CO_R", 0.0, 0.95, 0.40, 0.05)
mut_r_3 = st.sidebar.slider("Trial 3 - MUT_R", 0.01, 0.05, 0.05, 0.01)

with st.sidebar.expander("⚙️ Engine Settings"):
    pop_size = st.number_input("Population Size", 10, 500, 50, 10)
    generations = st.number_input("Generations", 10, 1000, 100, 10)

with st.expander("📊 View Audience Rating Matrix (program_ratings.csv)", expanded=False):
    st.dataframe(df, use_container_width=True)

# 5. Execute Optimization & Display Results
if st.button("🚀 Optimize Broadcast Lineup (Run 3 Trials)", type="primary", use_container_width=True):
    trials = [
        ("Trial 1", co_r_1, mut_r_1),
        ("Trial 2", co_r_2, mut_r_2),
        ("Trial 3", co_r_3, mut_r_3),
    ]

    results = []
    for idx, (name, co_val, mut_val) in enumerate(trials):
        sched, score = run_genetic_algorithm(co_val, mut_val, int(pop_size), int(generations), seed=42 + idx)
        slot_ratings = [ratings[sched[i]][time_slots[i]] for i in range(len(time_slots))]
        table_df = pd.DataFrame({
            "Time Slot": time_slots,
            "Scheduled Program": [format_program_label(p) for p in sched],
            "Rating": slot_ratings,
            "Status": [get_slot_status(r) for r in slot_ratings],
        })
        results.append({
            "name": name,
            "co_r": co_val,
            "mut_r": mut_val,
            "score": score,
            "raw_sched": sched,
            "slot_ratings": slot_ratings,
            "table": table_df,
        })

    # Top KPI Metrics Row
    winner = max(results, key=lambda x: x["score"])
    peak_idx = max(range(len(time_slots)), key=lambda i: winner["slot_ratings"][i])
    top_prog = pd.Series(winner["raw_sched"]).mode()[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🏆 Best Performing Trial", winner["name"], f"CO_R={winner['co_r']} | MUT_R={winner['mut_r']}")
    k2.metric("📈 Highest Total Rating", f"{winner['score']:.2f}", f"{len(time_slots)} Broadcast Hours")
    k3.metric("🔥 Peak Audience Slot", time_slots[peak_idx], f"Rating: {winner['slot_ratings'][peak_idx]:.2f}")
    k4.metric("⭐ Top Scheduled Show", format_program_label(top_prog))

    st.divider()

    # Organized Tabs
    tab1, tab2, tab3 = st.tabs([
        "📅 Generated Schedules (3 Trials)",
        "📈 Hourly Rating Comparison Chart",
        "📋 Summary & Export",
    ])

    with tab1:
        cols = st.columns(3)
        for idx, res in enumerate(results):
            with cols[idx]:
                st.subheader(f"{res['name']}")
                st.caption(f"**CO_R:** `{res['co_r']}` | **MUT_R:** `{res['mut_r']}` | **Total Rating:** `{res['score']:.2f}`")
                st.dataframe(res["table"], use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Hourly Audience Rating Across Trials")
        chart_df = pd.DataFrame(
            {res["name"]: res["slot_ratings"] for res in results},
            index=time_slots,
        )
        st.line_chart(chart_df, use_container_width=True)

    with tab3:
        st.subheader("Parameter Performance Summary")
        summary_df = pd.DataFrame([
            {
                "Trial": r["name"],
                "Crossover Rate (CO_R)": r["co_r"],
                "Mutation Rate (MUT_R)": r["mut_r"],
                "Total Schedule Rating": round(r["score"], 2),
            }
            for r in results
        ])
        st.table(summary_df)

        csv_export = winner["table"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download Winning Schedule ({winner['name']} CSV)",
            data=csv_export,
            file_name="primegrid_best_schedule.csv",
            mime="text/csv",
        )
