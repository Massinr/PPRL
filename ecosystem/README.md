# Predator-Prey RL Ecosystem (from scratch, NumPy only)
pip install pygame numpy matplotlib      # tkinter ships with Python (Linux: sudo apt install python3-tk)
python main.py                           # GUI: world + 5 windows (F1 Controls, F2 Data, F3 Audit, F4 Neural net, F6 Learning)
python headless_train.py --minutes 3 --sharks 3 --plot curve.png --save models/shark_policy.npz
python tests/test_all.py   (or pytest tests)
Keys: SPACE pause | +/- speed | S/s sharks +/-1 live | F/f fish +/-5 | A/a attack range | L learning on/off |
      B reset brain | R reset world | F5 save | F9 load (models/shark_policy.npz is a pre-trained example) | C console echo
Logs: logs/audit_*.csv (every event) ; console prints EAT/DEATH/UPDATE events + a table every 10 sim-seconds.
