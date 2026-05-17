# ANTIGRAVITY AI UPGRADE PLAN

## Baseline Inventory
- **Project Root**: `C:\Users\ali_z\ANU AI\LK20`
- **Current Working Directory**: `C:\Users\ali_z\ANU AI\LK20`
- **Python Version**: 3.12.4

### Presence of Critical Files
- [x] `lk20_server.py`
- [x] `lk20_main.py`
- [x] `lk20_kernel.py`
- [x] `local_ai_adapter.py`
- [x] `embedding_index.py`
- [x] `local_ai/local_ai.py`
- [x] `local_ai/semantic_attractors.py`
- [x] `local_ai/sentence_builder.py`
- [x] `local_ai/dictionary_lexicon_ingestor.py`
- [x] `twin_anything.py`
- [x] `local_ai/twin_any_language.py`
- [x] `tn.py` (Root)
- [x] `akkurat_atomtn_stack/tn.py`

## Verification Sequence Status
- [ ] `sdk_preflight.py`
- [ ] `sdk_contract_test.py`
- [ ] `privacy_boundary_test.py`
- [ ] `verify_network_integrity.py`
- [ ] `run_regression.py`
- [ ] `python local_ai\local_ai.py doctor --compile`
- [ ] `python lk20_kernel.py --mode status`
- [ ] `python lk20_main.py health`

## Upgrade Phases
1. **Verification**: Confirm baseline and file layout.
2. **TN Verification**: Ensure distinctness of both `tn.py` files.
3. **Semantic Bank Compatibility**: Fix `load_npz` in `semantic_attractors.py`.
4. **Embedding Hardening**: Upgrade `embedding_index.py` for safety.
5. **Adapter Upgrade**: Transform `local_ai_adapter.py` into a governed bridge.
6. **Kernel Integration**: Add `ai_status()` to `LK20MainApp`.
7. **API Wiring**: Wire `/api/ai/status` in `lk20_server.py`.
8. **Regression Runner**: Update `run_regression.py` with AI checks.
9. **Final Validation**: Server bring-up and smoke tests.
