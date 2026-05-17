import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\ali_z\ANU AI\LK20")
STACK_DIR = PROJECT_ROOT / "akkurat_atomtn_stack"
LOCAL_AI_DIR = PROJECT_ROOT / "local_ai"

print("--- TN Import Resolution Check ---")

# 1. Root Import
sys.path.insert(0, str(PROJECT_ROOT))
try:
    import tn
    print(f"Root 'tn' resolves to: {tn.__file__}")
    del sys.modules['tn']
except Exception as e:
    print(f"Root 'tn' import FAILED: {e}")
sys.path.pop(0)

# 2. Local AI Context
print("\n--- Local AI Doctor Context Check ---")
sys.path.insert(0, str(STACK_DIR))
sys.path.insert(0, str(LOCAL_AI_DIR))
try:
    # Simulating local_ai.py doctor imports
    import tn as stack_tn
    print(f"Stack 'tn' resolves to: {stack_tn.__file__}")
    
    import math_utils
    print(f"math_utils resolves to: {math_utils.__file__}")
    
    import geometry
    print(f"geometry resolves to: {geometry.__file__}")
    
    import ttn_state
    print(f"ttn_state resolves to: {ttn_state.__file__}")
except Exception as e:
    print(f"Stack/Local AI imports FAILED: {e}")
finally:
    sys.path.pop(0)
    sys.path.pop(0)

# 3. Kernel Check
print("\n--- LK20 Kernel Context ---")
sys.path.insert(0, str(PROJECT_ROOT))
try:
    import lk20_kernel
    # We check if lk20_kernel uses the root tn.py
    # (By checking its internal imports if possible or assuming based on path)
    print(f"lk20_kernel resolves to: {lk20_kernel.__file__}")
except Exception as e:
    print(f"lk20_kernel import FAILED: {e}")
sys.path.pop(0)

# 4. Language Twin Check
print("\n--- Language Twin Candidates ---")
candidates = ["twin_anything", "twin_any_language"]
for c in candidates:
    p = PROJECT_ROOT / f"{c}.py"
    lp = LOCAL_AI_DIR / f"{c}.py"
    if p.exists():
        print(f"  [FOUND] {p}")
    if lp.exists():
        print(f"  [FOUND] {lp}")
