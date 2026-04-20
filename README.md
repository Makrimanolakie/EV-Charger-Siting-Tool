# EV Charger Siting Tool

Εργαλείο γεωχωρικής ανάλυσης και υποστήριξης αποφάσεων για την επιλογή κατάλληλων τοποθεσιών εγκατάστασης φορτιστών ηλεκτρικών οχημάτων σε ιδιωτικούς χώρους στάθμευσης.

### Χαρακτηριστικά
- Επεξεργασία δεδομένων χώρων στάθμευσης (σούπερ μάρκετ, εμπορικά κέντρα, δημοτικά parking)
- Υπολογισμός καταλληλότητας θέσεων
- Οπτικοποίηση αποτελεσμάτων σε χάρτη
- Υποστήριξη πολλαπλών πόλεων (Αθήνα & Καβάλα)

### Εγκατάσταση
```bash
git clone https://github.com/USERNAME/ev-charger-siting-tool.git
cd ev-charger-siting-tool
pip install -r requirements.txt
python run_kavala.py
python run_athens.py

---

### 2. `requirements.txt`

```txt
numpy
pandas
matplotlib
scipy
scikit-learn
pulp
