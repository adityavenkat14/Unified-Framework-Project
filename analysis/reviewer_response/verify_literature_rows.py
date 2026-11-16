"""Re-checks web-extracted literature rows: the per-method average printed in each
paper must equal the mean of the 11 dataset columns. A transcription error in any
cell would (almost always) break this. Rows copied from the arXiv versions on 13 Nov 2026."""
rows = {
 "CLIP-LoRA Table 2 (ViT-B/16, 16-shot); cols ImageNet SUN Aircraft EuroSAT Cars Food Pets Flowers Caltech DTD UCF | Avg": {
  "CoOp (4)": ([71.5,74.6,40.1,83.5,79.1,85.1,92.4,96.4,95.5,69.2,81.9], 79.0),
  "CoOp (16)": ([71.9,74.9,43.2,85.0,82.9,84.2,92.0,96.8,95.8,69.7,83.1], 80.0),
  "CoCoOp": ([71.1,72.6,33.3,73.6,72.3,87.4,93.4,89.1,95.1,63.7,77.2], 75.4),
  "TIP-Adapter-F": ([73.4,76.0,44.6,85.9,82.3,86.8,92.6,96.2,95.7,70.8,83.9], 80.7),
  "CLIP-Adapter": ([69.8,74.2,34.2,71.4,74.0,87.1,92.3,92.9,94.9,59.4,80.2], 75.5),
  "PLOT++": ([72.6,76.0,46.7,92.0,84.6,87.1,93.6,97.6,96.0,71.4,85.3], 82.1),
  "KgCoOp": ([70.4,73.3,36.5,76.2,74.8,87.2,93.2,93.4,95.2,68.7,81.7], 77.3),
  "TaskRes": ([73.0,76.1,44.9,82.7,83.5,86.9,92.4,97.5,95.8,71.5,84.0], 80.8),
  "MaPLe": ([71.9,74.5,36.8,87.5,74.3,87.4,93.2,94.2,95.4,68.4,81.4], 78.6),
  "ProGrad": ([72.1,75.1,43.0,83.6,82.9,85.8,92.8,96.6,95.9,68.8,82.7], 79.9),
  "CLIP-LoRA": ([73.6,76.1,54.7,92.1,86.3,84.2,92.4,98.0,96.4,72.0,86.7], 83.0)},
 "GDA Table 9 (ViT-B/16, 16-shot); cols Pets Flowers FGVC DTD EuroSAT Cars Food SUN Cal UCF IN | Avg": {
  "Zero-Shot CLIP": ([89.21,71.34,24.72,44.39,47.60,65.32,86.06,62.50,92.94,66.75,66.73], 65.23),
  "CoOp": ([92.53,96.47,42.91,68.50,80.87,83.09,87.21,75.29,95.77,82.24,71.92], 79.71),
  "Tip-Adapter": ([91.54,94.41,39.48,65.68,76.58,75.44,86.47,71.85,95.10,77.94,70.46], 76.81),
  "GDA": ([93.73,97.92,50.33,71.26,89.19,82.63,87.27,75.87,95.79,84.09,72.24], 81.85)},
}
for table, ms in rows.items():
    print(table)
    for m, (v, avg) in ms.items():
        dp = 1 if isinstance(avg, float) and round(avg, 1) == avg and "GDA" not in table else 2
        got = round(sum(v) / len(v), dp)
        print(f"  {m:<16s} recomputed {got:<6} printed {avg:<6} {'OK' if abs(got - avg) < 10 ** -dp + 1e-9 else 'MISMATCH'}")
