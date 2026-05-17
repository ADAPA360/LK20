import sys
import time
import json
import numpy as np
from akkurat_atom_hybrid import build_hybrid_runtime, AkkuratAtomHybridConfig

def main():
    # Make sure stdout is unbuffered
    cfg = AkkuratAtomHybridConfig(
        runtime_id="live_telemetry",
        input_dim=3,
        control_dim=16,
        output_dim=16,
        atom_profile="fast",
        atom_dt=0.05,
        enable_atom=True,
        enable_tcfc=True,
    )

    try:
        rt = build_hybrid_runtime(cfg)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    step = 0
    rng = np.random.default_rng(42)

    while True:
        try:
            x = rng.normal(0, 1, size=cfg.input_dim).astype(np.float32)
            res = rt.step(x, dt=cfg.atom_dt)
            
            # extract basic things we want
            payload = {
                "ok": res.ok,
                "step": res.step,
                "ts": res.ts,
                "atom_features_norm": float(np.linalg.norm(res.atom_features)),
                "tcfc_state_norm": float(np.linalg.norm(res.tcfc_state)),
                "output_norm": float(np.linalg.norm(res.output)),
                "metrics": res.metrics,
                "elapsed_s": res.elapsed_s
            }
            print(json.dumps(payload), flush=True)
            time.sleep(1.0)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)
            time.sleep(5.0)

if __name__ == "__main__":
    main()
