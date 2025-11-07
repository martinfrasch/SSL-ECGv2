# Compute Options for SSL-ECG Training

## Your Dataset Specifications

- **ECGs:** 450 recordings
- **Duration:** 40 minutes each
- **Sampling rate:** 1000 Hz (will be downsampled to 256 Hz)
- **Total raw samples:** 1.08 billion
- **After preprocessing:** ~108,000 windows of 10 seconds each
- **Data size:** ~1.5-2 GB
- **Training time estimate:** 8-15 hours (single GPU)

---

## Option Comparison Table

| Feature | Colab Free | Colab Pro | GCP T4 | GCP V100 |
|---------|------------|-----------|---------|----------|
| **Cost** | Free | $10/month | ~$6-8 | ~$15-20 |
| **GPU** | T4 (16GB) | T4/V100 | T4 (16GB) | V100 (16GB) |
| **Time Limit** | 12 hours | 24 hours | Unlimited | Unlimited |
| **Reliability** | ⚠️ Unstable | ✅ Good | ✅ Excellent | ✅ Excellent |
| **Speed** | 12-15 hrs | 8-12 hrs | 12-15 hrs | 6-8 hrs |
| **RAM** | 12-13 GB | ~25 GB | 15 GB | 15 GB |
| **Checkpointing** | Manual | Auto | Auto | Auto |
| **Disconnection Risk** | ❌ High | ⚠️ Low | ✅ None | ✅ None |
| **Setup Time** | 5 min | 5 min | 15 min | 15 min |
| **Best For** | Testing | Small jobs | Production | Production |

---

## Detailed Analysis

### Option 1: Google Colab Free ⚠️ RISKY

**Estimated Runtime:** 12-15 hours
**Cost:** $0
**Success Probability:** ~50%

**Pros:**
- ✅ Zero cost
- ✅ Quick setup (5 minutes)
- ✅ Good for initial testing
- ✅ Tesla T4 GPU is sufficient

**Cons:**
- ❌ **12-hour hard limit** - YOUR JOB IS 12-15 HOURS
- ❌ Random disconnections even before 12 hours
- ❌ Must keep browser open
- ❌ No guaranteed GPU availability
- ❌ May need to restart and lose progress

**Recommendation:** ⚠️ **NOT RECOMMENDED** for your dataset size
- Too risky - 80% chance of timeout
- Only viable if you reduce to:
  - 15 epochs (instead of 30)
  - 3 folds (instead of 5)
  - Even then, still risky

**When to Use:**
- Quick tests with subset of data
- Debugging code
- Proof of concept with <5 hours runtime

---

### Option 2: Google Colab Pro ✅ SAFE BACKUP

**Estimated Runtime:** 8-12 hours
**Cost:** $10/month (cancel after 1 month)
**Success Probability:** ~95%

**Pros:**
- ✅ **24-hour timeout** (safe for your job)
- ✅ Better GPU allocation (often get V100)
- ✅ ~25GB RAM (more headroom)
- ✅ Background execution
- ✅ Priority access
- ✅ Easy setup (5 minutes)
- ✅ Can run multiple experiments in one month

**Cons:**
- 💰 $10/month subscription
- Still can disconnect (but rare)
- Not as fast as dedicated GCP

**Recommendation:** ✅ **GOOD BACKUP OPTION**
- Safe for your job (12 hours << 24-hour limit)
- Affordable
- Easy to set up
- Cancel after finishing experiments

**When to Use:**
- Don't want to deal with GCP setup
- Need results quickly
- Running 1-5 experiments
- Want simplicity over performance

---

### Option 3: GCP with Tesla T4 ✅ RELIABLE

**Estimated Runtime:** 12-15 hours
**Cost:** ~$6-8 per run (450+ runs with your $30k credits)
**Success Probability:** 99.9%

**Configuration:**
```
Machine: n1-standard-2
vCPUs: 2
RAM: 7.5 GB
GPU: 1x Tesla T4 (16GB VRAM)
Storage: 50 GB SSD
Cost: ~$0.50/hour
```

**Pros:**
- ✅ **No timeout** - run for days if needed
- ✅ **100% reliable** - no random disconnections
- ✅ Can run unattended (close laptop, go home)
- ✅ Production-grade infrastructure
- ✅ Easy to monitor remotely
- ✅ ~450+ full experiments with your credits
- ✅ Can parallelize multiple experiments

**Cons:**
- ⏱️ 15 minutes setup time
- Requires GCP account setup
- Need to monitor costs (but you have plenty)

**Recommendation:** ✅ **BEST VALUE FOR RELIABILITY**
- Perfect balance of cost and reliability
- Same GPU as Colab Free, but no timeout
- Can run overnight worry-free
- Only ~$6-8 for complete experiment

**When to Use:**
- Want guaranteed completion
- Need to run overnight/weekend
- Running multiple experiments
- Professional/publication-quality work

---

### Option 4: GCP with Tesla V100 🚀 FASTEST

**Estimated Runtime:** 6-8 hours (2x faster than T4)
**Cost:** ~$15-20 per run (150+ runs with your $30k credits)
**Success Probability:** 99.9%

**Configuration:**
```
Machine: n1-standard-4
vCPUs: 4
RAM: 15 GB
GPU: 1x Tesla V100 (16GB VRAM)
Storage: 100 GB SSD
Cost: ~$2.50/hour
```

**Pros:**
- ✅ **2x faster** than T4
- ✅ All benefits of GCP (no timeout, reliable, unattended)
- ✅ More VRAM headroom (16GB vs 16GB but faster)
- ✅ Better for future larger datasets
- ✅ ~150+ experiments with your credits
- ✅ Can run multiple in parallel

**Cons:**
- 💰 Higher cost (~$15-20 vs ~$6-8)
- Same setup time as T4

**Recommendation:** 🚀 **BEST FOR SPEED**
- Finishes in single workday
- Worth the extra $10 for time savings
- Still excellent value with your credits
- Run during work hours, done by evening

**When to Use:**
- Need results fast
- Running many experiments
- Want to iterate quickly
- Time is more valuable than $10-15

---

## 🎯 My Specific Recommendation for Your Situation

Given your needs:
- ✅ 450 ECGs (~108K windows)
- ✅ $30k in GCP credits
- ✅ Need publication-quality results
- ✅ May want to run multiple experiments
- ✅ Have new prospective dataset to analyze

### Primary Recommendation: **GCP with Tesla V100** 🏆

**Why:**
1. **Speed:** Done in 6-8 hours (half a workday)
2. **Reliability:** No timeout, no disconnections
3. **Cost:** Only $15-20, you have $30k (can run 1,500+ times)
4. **Scalability:** Easy to run multiple experiments in parallel
5. **Professional:** Suitable for manuscript-quality results
6. **Future-proof:** Fast enough for larger prospective dataset

**Setup:**
```bash
# On your local machine:
bash scripts/setup_gcp_vm.sh ssl-ecg-vm us-central1-a nvidia-tesla-v100
```

### Backup Option: **Colab Pro** 📱

If you want to test first or avoid GCP setup:
1. Subscribe to Colab Pro ($10)
2. Upload `notebooks/SSL_ECG_Colab_Runner.ipynb`
3. Run the notebook
4. Wait 8-12 hours
5. Cancel subscription after finishing

---

## Cost-Benefit Analysis

### For Your Dataset (450 ECGs):

| Option | Time | Cost | Risk | Effort |
|--------|------|------|------|--------|
| Colab Free | 12-15h | $0 | HIGH ❌ | Low |
| Colab Pro | 8-12h | $10 | Low ⚠️ | Low |
| GCP T4 | 12-15h | $6-8 | None ✅ | Medium |
| GCP V100 | 6-8h | $15-20 | None ✅ | Medium |

### For Multiple Experiments (5 seeds × 5 folds):

| Option | Total Time | Total Cost | Reliability |
|--------|------------|------------|-------------|
| Colab Free | 60-75h | $0 | 10% success ❌ |
| Colab Pro | 40-60h | $10-20 | 80% success ⚠️ |
| GCP T4 | 60-75h | $30-40 | 100% ✅ |
| GCP V100 | 30-40h | $75-100 | 100% ✅ |

**Your $30k credits can cover:**
- T4: ~4,000 full experiments
- V100: ~1,500 full experiments

---

## Setup Instructions

### GCP Setup (Recommended)

1. **One-time setup (15 minutes):**
   ```bash
   # Install gcloud CLI if not installed
   # https://cloud.google.com/sdk/docs/install

   # Authenticate
   gcloud auth login

   # Set project
   gcloud config set project YOUR_PROJECT_ID

   # Enable APIs
   gcloud services enable compute.googleapis.com
   ```

2. **Create VM with V100:**
   ```bash
   bash scripts/setup_gcp_vm.sh ssl-ecg-vm us-central1-a nvidia-tesla-v100
   ```

3. **Upload your data:**
   ```bash
   gcloud compute scp /local/path/to/felicitys_mecg_0.npy ssl-ecg-vm:~/data/ --zone=us-central1-a
   ```

4. **Upload your code:**
   ```bash
   git push origin claude/review-and-plan-improvements-011CUtwG7jQk8UkVsFwcjUs3

   # On VM:
   gcloud compute ssh ssl-ecg-vm --zone=us-central1-a
   git clone https://github.com/martinfrasch/SSL-ECGv2.git
   cd SSL-ECGv2
   git checkout claude/review-and-plan-improvements-011CUtwG7jQk8UkVsFwcjUs3
   ```

5. **Run training:**
   ```bash
   cd codes
   nohup python3 train_fixed.py --random_seed 42 > ../training.log 2>&1 &
   ```

6. **Monitor progress:**
   ```bash
   tail -f training.log
   ```

7. **When done:**
   ```bash
   # Download results
   gcloud compute scp ssl-ecg-vm:~/SSL-ECGv2/output . --recurse --zone=us-central1-a

   # Stop VM (saves money)
   gcloud compute instances stop ssl-ecg-vm --zone=us-central1-a

   # Delete when completely done
   gcloud compute instances delete ssl-ecg-vm --zone=us-central1-a
   ```

### Colab Pro Setup (Backup)

1. **Subscribe:** https://colab.research.google.com/signup
2. **Open notebook:** Upload `notebooks/SSL_ECG_Colab_Runner.ipynb`
3. **Enable GPU:** Runtime → Change runtime type → GPU
4. **Follow notebook instructions**
5. **Keep tab open** (or enable background execution)

---

## Monitoring and Progress Tracking

### GCP:
```bash
# SSH and monitor
gcloud compute ssh ssl-ecg-vm --zone=us-central1-a
tail -f SSL-ECGv2/training.log

# Or use TensorBoard
tensorboard --logdir=SSL-ECGv2/summaries --host=0.0.0.0 --port=6006

# Access from local machine:
gcloud compute ssh ssl-ecg-vm --zone=us-central1-a -- -L 6006:localhost:6006
# Then open http://localhost:6006
```

### Colab:
- TensorBoard built into notebook
- Progress bars visible in output
- Can check output files periodically

---

## Expected Timeline

### Single Experiment:

| Stage | GCP V100 | GCP T4 | Colab Pro |
|-------|----------|---------|-----------|
| Setup | 15 min | 15 min | 5 min |
| SSL Training | 4-5h | 8-10h | 6-8h |
| Downstream Tasks | 2-3h | 4-5h | 2-4h |
| **Total** | **~7-8h** | **~13-15h** | **~9-12h** |

### Multiple Seeds (5):

| Option | Total Time | When Done |
|--------|------------|-----------|
| GCP V100 (parallel) | 7-8h | Same day |
| GCP V100 (serial) | 35-40h | 2 days |
| GCP T4 (parallel) | 13-15h | Same day |
| GCP T4 (serial) | 65-75h | 3 days |
| Colab Pro (serial) | 45-60h | 3 days |

**With $30k credits, you can run 5 seeds in parallel on 5 V100 VMs:**
- Cost: 5 × $20 = $100
- Time: 8 hours
- Done in 1 day!

---

## Final Recommendation

**For your specific situation:**

1. **Best option: GCP with Tesla V100** 🥇
   - Start: Today afternoon
   - Done: Tonight/tomorrow morning
   - Cost: $15-20
   - Quality: Production-grade

2. **Budget option: GCP with Tesla T4** 🥈
   - Start: Today afternoon
   - Done: Tomorrow evening
   - Cost: $6-8
   - Quality: Production-grade

3. **Easy option: Colab Pro** 🥉
   - Start: In 5 minutes
   - Done: Tomorrow
   - Cost: $10/month
   - Quality: Good (but less reliable)

4. **Avoid: Colab Free** ❌
   - Too risky for your dataset size
   - 80% chance of timeout
   - Not worth the frustration

**My pick: GCP V100 - Worth the extra $10-15 for peace of mind and speed.**

---

## Questions?

- **GCP Setup:** See `scripts/setup_gcp_vm.sh`
- **Colab Usage:** See `notebooks/SSL_ECG_Colab_Runner.ipynb`
- **Training Details:** See `SETUP.md`
- **Code Questions:** See `CODE_REVIEW.md`

---

**Ready to start? Let me know which option you choose and I can provide more detailed guidance!**
