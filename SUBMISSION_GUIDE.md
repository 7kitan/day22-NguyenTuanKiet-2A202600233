# Day 22 Lab — Submission Guide

Complete submission checklist and step-by-step instructions.

---

## ⚡ Quick Start: Everything is Automatic!

**You don't need to do anything manually in LangSmith console.**

The scripts handle everything:
- ✅ Sending traces to LangSmith
- ✅ Pushing prompts to Prompt Hub
- ✅ Running evaluations
- ✅ Saving logs to evidence files

**Your only job:** Run the code, then screenshot the results.

---

## Pre-Execution Checklist

Before running anything, verify:

- [ ] `.env` file has valid `OPENAI_API_KEY`, `LANGCHAIN_API_KEY`, `OPENAI_BASE_URL`
- [ ] GitHub repo is initialized and remote is set
- [ ] `.gitignore` excludes `.env` and `__pycache__/`
- [ ] `evidence/` directory exists and is empty
- [ ] `data/` directory exists

---

## Phase 1: Run the Lab (25 minutes)

### ✨ Everything is Automatic!

**You don't need to do anything in LangSmith console.**

The scripts automatically:
- Send traces to LangSmith (Step 1 & 2)
- Push prompts to Prompt Hub (Step 2)
- Run evaluations (Step 3)
- Save logs to evidence files (Steps 2 & 4)

### Just Run This:
```bash
python run_all.py
```

**That's it!** The script will:
1. Build RAG pipeline and send 50 traces
2. Push 2 prompts to Prompt Hub and send 50 more traces
3. Run RAGAS evaluation (~20 min)
4. Run Guardrails validators and auto-save logs

**Estimated time:** ~25 minutes (Step 3 RAGAS is slow)

---

## Phase 2: Collect Evidence (After Running)

### What's Already Done ✅
After `python run_all.py` completes:
- ✅ 100+ traces sent to LangSmith (automatic)
- ✅ 2 prompts pushed to Prompt Hub (automatic)
- ✅ RAGAS evaluation complete (automatic)
- ✅ Logs saved to `evidence/02_ab_routing_log.txt`, `evidence/04_pii_demo_log.txt`, `evidence/04_json_demo_log.txt` (automatic)

### What You Need to Do: Screenshot 3 Things

#### 1. LangSmith Traces (Step 1 & 2)

**File:** `evidence/01_langsmith_traces.png`

**Steps:**
1. Open https://smith.langchain.com
2. Navigate to your project
3. Click the **Run** tab
4. You should see ≥100 traces (50 from step 1 + 50 from step 2)
5. Take a screenshot

**What to show in screenshot:**
- Project name
- Run tab selected
- Multiple traces listed with timestamps
- At least 100 traces visible

---

#### 2. Prompt Hub (Step 2)

**File:** `evidence/02_prompt_hub.png`

**Steps:**
1. Open https://smith.langchain.com
2. Click the **Prompt Hub** tab
3. You should see both prompts (automatically pushed by the script):
   - `my-rag-prompt-v1` (V1 – concise answers)
   - `my-rag-prompt-v2` (V2 – structured answers)
4. Take a screenshot

**What to show in screenshot:**
- Prompt Hub tab selected
- Both prompt names visible
- Version numbers visible

---

#### 3. RAGAS Scores (Step 3)

**File:** `evidence/03_ragas_scores.png`

**Steps:**
1. Look at the console output from `python run_all.py`
2. Find the "Comparison: V1 vs V2" section
3. Screenshot the comparison table showing:
   - Metric names (faithfulness, answer_relevancy, context_recall, context_precision)
   - V1 and V2 scores for each metric
   - Winner indicators (← V1 or ← V2)
   - "✅ Target met" or "⚠️ Below target" line

**What to show in screenshot:**
- The full comparison table
- All 4 metrics with scores
- Target met confirmation

---

## Phase 3: Prepare for Submission

### Step 1: Verify all evidence files exist
```bash
ls -la evidence/
```

**Expected output:**
```
01_langsmith_traces.png          ← You screenshot this
02_prompt_hub.png                ← You screenshot this
02_ab_routing_log.txt            ← Auto-saved by script
03_ragas_scores.png              ← You screenshot this
03_ragas_report.json             ← You copy this
04_pii_demo_log.txt              ← Auto-saved by script
04_json_demo_log.txt             ← Auto-saved by script
```

### Step 2: Copy RAGAS report
```bash
cp data/ragas_report.json evidence/03_ragas_report.json
```

### Step 3: Verify .env is NOT in git
```bash
git status
```

**Should NOT show `.env` in the output**

If `.env` is staged, remove it:
```bash
git rm --cached .env
```

---

## Phase 4: Git Commit & Push

### Stage all files
```bash
git add .
```

### Verify staging (should NOT include .env)
```bash
git status
```

### Commit
```bash
git commit -m "Day22 lab submission: LangSmith + Prompt Versioning + RAGAS + Guardrails"
```

### Push to GitHub
```bash
git push origin main
```

---

## Phase 5: Submit on Course Portal

Provide the following information:

1. **GitHub Repository URL**
   - Format: `https://github.com/yourname/day22-langsmith-lab`
   - Verify it's public and contains all files

2. **LangSmith Project URL**
   - Format: `https://smith.langchain.com/o/<org-id>/projects/p/<project-id>`
   - Find this in your LangSmith dashboard

3. **Confirmation**
   - Confirm `evidence/` folder exists in repo with all 7 files
   - Confirm `.env` is NOT committed

---

## Verification Checklist

### Before Submission
- [ ] All 4 steps ran successfully
- [ ] `evidence/` folder has 7 files
- [ ] `data/ragas_report.json` exists
- [ ] `.env` is in `.gitignore` and NOT committed
- [ ] Git history shows clean commit
- [ ] GitHub repo is public and accessible

### Evidence Files
- [ ] `01_langsmith_traces.png` — ≥50 traces visible
- [ ] `02_prompt_hub.png` — both prompts visible
- [ ] `02_ab_routing_log.txt` — routing decisions logged
- [ ] `03_ragas_scores.png` — comparison table visible
- [ ] `03_ragas_report.json` — valid JSON with scores
- [ ] `04_pii_demo_log.txt` — 6 PII test cases
- [ ] `04_json_demo_log.txt` — 5 JSON test cases

### Deliverables Met
- [ ] ≥100 LangSmith traces (50 from step 1 + 50 from step 2)
- [ ] 2 prompts in Prompt Hub
- [ ] A/B routing working (50/50 split)
- [ ] RAGAS evaluation complete
- [ ] Faithfulness ≥ 0.8 (or documented attempt)
- [ ] Guardrails validators working (PII + JSON)

---

## Common Issues & Solutions

### Issue: `.env` committed to git
**Solution:**
```bash
git rm --cached .env
git commit -m "Remove .env from tracking"
git push origin main
```

### Issue: Missing API keys
**Solution:**
1. Check `.env` file has all 3 keys:
   - `OPENAI_API_KEY`
   - `LANGCHAIN_API_KEY`
   - `OPENAI_BASE_URL`
2. Verify keys are valid (not placeholder text)

### Issue: Step 3 takes too long
**Solution:** This is normal. RAGAS evaluation makes many LLM calls (~20 min is expected).

### Issue: Faithfulness < 0.8
**Solution:** Try one of:
- Reduce `chunk_size` in vectorstore (e.g., 300 instead of 500)
- Improve prompt instructions
- Ensure knowledge base has relevant content

### Issue: Screenshots not captured
**Solution:** Use platform-specific tools:
- **Mac:** `Cmd+Shift+4` (drag to select area)
- **Windows:** `Win+Shift+S` (snip tool)
- **Linux:** `gnome-screenshot -a` (area selection)

### Issue: Traces not appearing in LangSmith
**Solution:**
1. Verify `LANGCHAIN_TRACING_V2=true` in `.env`
2. Verify `LANGCHAIN_API_KEY` is valid
3. Verify `LANGCHAIN_PROJECT` name is set
4. Wait 30 seconds for traces to sync
5. Refresh LangSmith UI

---

## Quick Reference Commands

```bash
# Run all steps (auto-logs steps 2 & 4)
python run_all.py

# Run specific step
python run_all.py --step 3

# Run individual steps (auto-logs steps 2 & 4)
python 01_langsmith_rag_pipeline.py
python 02_prompt_hub_ab_routing.py          # Auto-saves to evidence/02_ab_routing_log.txt
python 03_ragas_evaluation.py
python 04_guardrails_validator.py           # Auto-saves to evidence/04_pii_demo_log.txt + 04_json_demo_log.txt

# Copy RAGAS report
cp data/ragas_report.json evidence/03_ragas_report.json

# Verify evidence
ls -la evidence/

# Git workflow
git add .
git status
git commit -m "Day22 lab submission"
git push origin main
```

---

## Timeline

| Phase | Time | Action |
|-------|------|--------|
| Pre-execution | 5 min | Verify setup |
| Step 1 | 2 min | Run RAG pipeline |
| Step 2 | 2 min | Run Prompt Hub + A/B routing |
| Step 3 | 20 min | Run RAGAS evaluation |
| Step 4 | 1 min | Run Guardrails validators |
| Evidence collection | 10 min | Screenshots + logs |
| Git + submission | 5 min | Commit and push |
| **Total** | **~45 min** | Complete submission |

---

## Support

If you encounter issues:
1. Check the error message carefully
2. Review the "Common Issues & Solutions" section above
3. Verify all prerequisites are met
4. Check that API keys are valid and not expired

Good luck! 🚀
