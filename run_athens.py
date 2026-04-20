from ev_optimizer import load_data, optimize_chargers, plot_results

df = load_data("athens")
K, optimal_locs, selected = optimize_chargers(df, "athens")
print(f"K = {K}")
print(selected[['name', 'lat', 'lon', 'capacity', 'grade']])
plot_results(df, optimal_locs, K, "Αθήνα")
