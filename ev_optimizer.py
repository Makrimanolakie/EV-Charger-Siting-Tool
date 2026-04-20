import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, value, PULP_CBC_CMD

def load_data(city="kavala"):
    if city.lower() == "kavala":
        data = [
            {'name': 'Δημοτική Αγορά Καβάλας (Αγ. Γεωργίου)', 'lat': 40.9365, 'lon': 24.4068, 'capacity': 250, 'grade': 5},
            {'name': 'Σκλαβενίτης Καβάλα (κεντρικό)', 'lat': 40.9402, 'lon': 24.4015, 'capacity': 180, 'grade': 4},
            {'name': 'ΑΒ Βασιλόπουλος Καβάλα (Ελευθερίας)', 'lat': 40.9378, 'lon': 24.3982, 'capacity': 150, 'grade': 4},
            {'name': 'Lidl Καβάλα (Πτολεμαίων)', 'lat': 40.9421, 'lon': 24.4123, 'capacity': 120, 'grade': 3},
            {'name': 'My Market Καβάλα (Περιφερειακό)', 'lat': 40.9350, 'lon': 24.4200, 'capacity': 200, 'grade': 4},
            {'name': 'Parking Ροδόπης', 'lat': 40.9370, 'lon': 24.4045, 'capacity': 100, 'grade': 4},
            {'name': 'Parking Αγ. Ιωάννη', 'lat': 40.9360, 'lon': 24.4070, 'capacity': 80, 'grade': 4},
            {'name': 'Parking Δημαρχίας', 'lat': 40.9368, 'lon': 24.4052, 'capacity': 120, 'grade': 4},
            {'name': 'Parking ΝΟΚ', 'lat': 40.9335, 'lon': 24.4100, 'capacity': 150, 'grade': 4},
            {'name': 'Parking Περιστερίου', 'lat': 40.9318, 'lon': 24.4127, 'capacity': 200, 'grade': 3},
            {'name': 'Parking Θάσου', 'lat': 40.9320, 'lon': 24.4150, 'capacity': 100, 'grade': 3},
            {'name': 'Parking Κέας', 'lat': 40.9355, 'lon': 24.4060, 'capacity': 80, 'grade': 3},
            {'name': 'Parking Ελευθερίας', 'lat': 40.9380, 'lon': 24.4005, 'capacity': 90, 'grade': 4},
            {'name': 'Parking Venizelou & Velissariou', 'lat': 40.9362, 'lon': 24.4075, 'capacity': 70, 'grade': 3},
            {'name': 'Parking Εθνικής Αντιστάσεως', 'lat': 40.9340, 'lon': 24.4110, 'capacity': 110, 'grade': 3},
            {'name': 'Parking Chrysostomou Smyrnis', 'lat': 40.9310, 'lon': 24.4130, 'capacity': 150, 'grade': 3},
            {'name': 'Parking Free National Resistance St.', 'lat': 40.9350, 'lon': 24.4050, 'capacity': 60, 'grade': 2},
        ]
    else:  # Athens sample
        data = [
            {'name': 'The Mall Athens (Marousi)', 'lat': 38.0402, 'lon': 23.7880, 'capacity': 1500, 'grade': 5},
            {'name': 'Golden Hall (Marousi)', 'lat': 38.0340, 'lon': 23.7930, 'capacity': 1200, 'grade': 5},
            {'name': 'River West (Egaleo)', 'lat': 37.9840, 'lon': 23.6790, 'capacity': 1000, 'grade': 5},
            {'name': 'Syntagma Underground', 'lat': 37.9756, 'lon': 23.7345, 'capacity': 850, 'grade': 5},
            {'name': 'Acropolis Parking', 'lat': 37.9690, 'lon': 23.7270, 'capacity': 400, 'grade': 5},
        ]
    return pd.DataFrame(data)

def optimize_chargers(df, city="kavala", min_dist_km=0.5):
    locations = df[['lat', 'lon']].to_numpy()
    demands = df['capacity'].to_numpy() * df['grade'].to_numpy()
    
    # Υπολογισμός K
    def find_optimal_k(data, max_k=10):
        distortions = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(data).inertia_ for k in range(1, max_k+1)]
        elbow = np.argmin(np.diff(distortions)) + 2
        return max(2, elbow)
    
    k_cluster = find_optimal_k(locations)
    ev_stock = 400 if city.lower() == "kavala" else 16000
    k_ev = max(2, int(ev_stock / 12))
    k_parking = int(df['capacity'].sum() / 20)
    K = min(k_cluster, max(k_ev, k_parking // 5))
    
    # PuLP Optimization
    N = len(locations)
    dist_matrix = cdist(locations, locations, metric='euclidean')
    
    prob = LpProblem("EV_Chargers", LpMinimize)
    y = {j: LpVariable(f"y_{j}", cat=LpBinary) for j in range(N)}
    x = {(i,j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i in range(N) for j in range(N)}
    
    prob += lpSum(demands[i] * dist_matrix[i,j] * x[(i,j)] for i in range(N) for j in range(N))
    
    for i in range(N):
        prob += lpSum(x[(i,j)] for j in range(N)) == 1
    for i in range(N):
        for j in range(N):
            prob += x[(i,j)] <= y[j]
    prob += lpSum(y[j] for j in range(N)) == K
    
    # Min distance constraint
    for j1 in range(N):
        for j2 in range(j1+1, N):
            if dist_matrix[j1, j2] < min_dist_km:
                prob += y[j1] + y[j2] <= 1
    
    prob.solve(PULP_CBC_CMD(msg=0))
    
    optimal_idx = [j for j in range(N) if value(y[j]) > 0.5]
    optimal_locs = locations[optimal_idx]
    selected = df.iloc[optimal_idx].copy()
    
    return K, optimal_locs, selected

def plot_results(df, optimal_locs, K, city="kavala"):
    locations = df[['lat', 'lon']].to_numpy()
    plt.figure(figsize=(10, 8))
    plt.scatter(locations[:,1], locations[:,0], c='red', s=50, label='Parking Spots')
    plt.scatter(optimal_locs[:,1], optimal_locs[:,0], c='lime', s=300, marker='*', label=f'Βέλτιστοι Φορτιστές (K={K})')
    plt.title(f'Βέλτιστη Τοποθέτηση Φορτιστών EV – {city.capitalize()}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ====================== Main ======================
if __name__ == "__main__":
    print("=== EV Charger Siting Tool ===\n")
    city = input("Επιλέξτε πόλη (kavala / athens): ").strip().lower() or "kavala"
    
    df = load_data(city)
    print(f"\nΦόρτωση {len(df)} σημείων για {city.capitalize()}")
    
    K, optimal_locs, selected = optimize_chargers(df, city)
    
    print(f"\nΠροτεινόμενος αριθμός φορτιστών: {K}")
    print("\nΠροτεινόμενες θέσεις:")
    print(selected[['name', 'lat', 'lon', 'capacity', 'grade']])
    
    plot_results(df, optimal_locs, K, city)
