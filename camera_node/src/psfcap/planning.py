"""Distance plan: which source-to-sensor distances to capture."""

from __future__ import annotations

import numpy as np


def plan_distances(args) -> list[float]:
    if args.distances:
        d = [float(x) for x in args.distances.split(",") if x.strip()]
    else:
        if args.stop is None:
            raise SystemExit("Specify --stop (and optionally --start/--num/--spacing) or --distances")
        a, b, n = args.start, args.stop, args.num
        if n < 2:
            d = [a]
        elif args.spacing == "inverse":
            d = list(1.0 / np.linspace(1.0 / a, 1.0 / b, n))
        else:
            d = list(np.linspace(a, b, n))
    # snap to motor steps, sort outward (consistent backlash), dedupe
    spm = args.steps_per_mm
    snapped = sorted({round((x - args.zero_distance) * spm) for x in d})
    out = [args.zero_distance + s / spm for s in snapped]
    for x in out:
        if x < args.zero_distance - 1e-9:
            raise SystemExit(f"Distance {x:.3f} mm is closer than the zero position "
                             f"({args.zero_distance} mm)")
        if x - args.zero_distance > args.max_travel + 1e-9:
            raise SystemExit(f"Distance {x:.3f} mm is beyond the maximum travel: the farthest "
                             f"allowed is {args.zero_distance + args.max_travel:.3f} mm "
                             f"(zero {args.zero_distance} + {args.max_travel} mm travel)")
    return out
