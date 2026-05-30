"""Compare buggy vs fixed training runs at comparable step counts."""
import csv, sys

def analyze(path, label):
    with open(path) as f:
        rows = list(csv.reader(f))
    data = [[float(x) for x in row] for row in rows]

    print(f"\n=== {label} ===")
    print(f"Samples: {len(data)}")

    for col in range(5):
        vals = [d[col] for d in data]
        n = max(len(vals) // 10, 1)
        start_avg = sum(vals[:n]) / n
        end_avg = sum(vals[-n:]) / n
        trend = "DOWN" if end_avg < start_avg else "UP"

        start10 = sum(vals[:n]) / n
        mid = sum(vals[len(vals)//2 - n//2:len(vals)//2 + n//2]) / n if len(vals) > n else start10
        end10 = sum(vals[-n:]) / n

        print(f"  Col{col}: {start10:.4f} -> {mid:.4f} -> {end10:.4f} [{trend}]  "
              f"min={min(vals):.4f}  max={max(vals):.4f}")

analyze(sys.argv[1], "RUN 1")
analyze(sys.argv[2], "RUN 2")
