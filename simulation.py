import simpy
import pandas as pd
import statistics
import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("foodcity_dataset.csv")

# Convert ArrivalTime to minutes
def time_to_minutes(t):
    hour, minute = map(int, t.split(":"))
    return (hour * 60) + minute

df["ArrivalMin"] = df["ArrivalTime"].apply(time_to_minutes)
df["ServiceMin"] = pd.to_numeric(df["ServiceMin"], errors="coerce")

# Clean data
df = df.dropna(subset=["ArrivalMin", "ServiceMin"])
df = df[df["ServiceMin"] > 0]

# Identify days
days = sorted(df["Date"].unique())

# -----------------------------
# Simulation Model
# -----------------------------
class FoodCitySimulation:
    def __init__(self, env, num_counters):
        self.env = env
        self.counters = simpy.Resource(env, capacity=num_counters)
        self.wait_times = []
        self.total_busy_time = 0.0

        # Bottleneck tracking
        self.queue_lengths = []
        self.queue_times = []

    def customer(self, arrival, service_time):
        yield self.env.timeout(arrival)
        arrive_time = self.env.now

        # Record queue length at arrival
        self.queue_lengths.append(len(self.counters.queue))
        self.queue_times.append(self.env.now)

        with self.counters.request() as req:
            yield req
            wait = self.env.now - arrive_time
            self.wait_times.append(wait)

            yield self.env.timeout(service_time)
            self.total_busy_time += service_time

# -----------------------------
# Run simulation for one day
# -----------------------------
def run_day_simulation(day_df, num_counters=4):
    if day_df.empty:
        return None, 0.0, 0

    day_df = day_df.sort_values("ArrivalMin").reset_index(drop=True)

    # Shift arrivals so first arrival = 0
    first_arrival = day_df["ArrivalMin"].min()
    day_df["AdjArrival"] = day_df["ArrivalMin"] - first_arrival

    env = simpy.Environment()
    sim = FoodCitySimulation(env, num_counters)

    for _, row in day_df.iterrows():
        env.process(sim.customer(row["AdjArrival"], row["ServiceMin"]))

    simulation_end = day_df["AdjArrival"].max() + day_df["ServiceMin"].max() + 10
    env.run(until=simulation_end)

    utilization = sim.total_busy_time / (num_counters * simulation_end)

    return sim, utilization, len(sim.wait_times)

# -----------------------------
# MAIN
# -----------------------------
NUM_COUNTERS = 4
all_waits = []
daily_utils = []

print("\n=== FOOD CITY SUPERMARKET QUEUE SIMULATION ===")
print(f"Counters: {NUM_COUNTERS}\n")

for day in days:
    day_df = df[df["Date"] == day]

    sim, util, count = run_day_simulation(day_df, NUM_COUNTERS)
    if sim is None:
        continue

    waits = sim.wait_times
    avg_wait = statistics.mean(waits)
    max_wait = max(waits)

    all_waits.extend(waits)
    daily_utils.append(util)

    print(f"Day: {day}")
    print(f" Customers served : {count}")
    print(f" Avg waiting time: {avg_wait:.2f} min")
    print(f" Max waiting time: {max_wait:.2f} min")
    print(f" Utilization     : {util:.2%}")
    print("-" * 40)

    # Waiting time histogram
    plt.figure(figsize=(7, 5))
    plt.hist(waits, bins=15, edgecolor="black")
    plt.title(f"Waiting Time Distribution ({day})")
    plt.xlabel("Waiting Time (minutes)")
    plt.ylabel("Customers")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"wait_time_{day.replace('/', '_')}.png")
    plt.close()

    # Bottleneck chart (queue length vs time)
    plt.figure(figsize=(7, 5))
    plt.plot(sim.queue_times, sim.queue_lengths)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Queue Length")
    plt.title(f"Queue Length Over Time (Bottleneck) - {day}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"queue_length_{day.replace('/', '_')}.png")
    plt.close()

# -----------------------------
# Overall Results
# -----------------------------
print("\n=== OVERALL PERFORMANCE ===")
print(f"Total customers  : {len(all_waits)}")
print(f"Overall avg wait : {statistics.mean(all_waits):.2f} min")
print(f"Overall max wait : {max(all_waits):.2f} min")
print(f"Avg utilization  : {statistics.mean(daily_utils):.2%}")

# Overall waiting time
plt.figure(figsize=(7, 5))
plt.hist(all_waits, bins=20, edgecolor="black")
plt.title("Overall Waiting Time Distribution")
plt.xlabel("Waiting Time (minutes)")
plt.ylabel("Customers")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("overall_waiting_time.png")
plt.close()

# Server utilization bar chart
plt.figure(figsize=(6, 5))
plt.bar(["Checkout Counters"], [statistics.mean(daily_utils)])
plt.ylim(0, 1)
plt.ylabel("Utilization")
plt.title("Average Server Utilization (Bottleneck Indicator)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("server_utilization_bar.png")
plt.close()
