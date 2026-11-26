# Finding and measuring cracks in concrete radiation shielding with a U-Net ensemble

## Summary

A simulated gamma-ray transmission scan of a 5 cm concrete wall gives a 64 × 64 flux image behind the wall. An ensemble of five small U-Nets turns that image into a crack-probability map. The crack's position, length and orientation are then measured from the map, and a single crack score decides whether the wall is cracked at all.

On the held-out test set (90 walls, 43 cracked):

- **Measurement.** Of the cracks found, the centre is placed to within 1.2 cm (x) and 1.5 cm (y) on average, length to within 1.85 cm and angle to within 10.8°. Always guessing the training average gives 8.4 cm, 7.0 cm, 5.3 cm and 41°.
- **Detection.** With the cut-off fixed on validation data, the ensemble finds 27 of 43 cracks (63%) at a 2% false-alarm rate, and 30 of 43 (70%) at 9%.

The main flaws are covered in section 4:

- **Cracks out of view.** The data generator places some cracks partly or wholly outside the detector's field of view, and these count as misses.
- **Inflated probabilities.** The network's outputs are pushed upwards, so the obvious 0.5 cut-off flags 45% of clean walls.
- **Small evaluation sets.** Results move noticeably between data splits.
- **Idealised physics.** The simulated detector is idealised, including an "uncollided" input channel that no real detector could separate out.

---

## 1. Simulation

All data come from OpenMC photon-transport simulations.

| Item | Value |
|---|---|
| Source | Cs-137 point source at the origin: 661.657 keV, isotropic, 10⁹ Bq × 0.851 γ per decay |
| Shield | Ordinary concrete, 2.3 g/cm³, 100 × 100 cm, 5 cm thick, front face at z = 27.5 cm, back face at z = 32.5 cm |
| Detector | 1 cm air layer at z = 37.5–38.5 cm, 64 × 64 mesh over 40 × 40 cm (0.625 cm pixels) |
| Statistics | 10 batches × 4 × 10⁶ photons per wall |

Three flux maps are saved for each wall, in photons cm⁻² s⁻¹:

- **Peak:** 600 keV up to the line. This covers small-angle scatter plus uncollided photons.
- **Scattered:** below 600 keV.
- **Uncollided:** within ±100 eV of the 661.657 keV line.

**Cracks.** Half the walls contain one crack. It's an air-filled rectangular slot, open at the back face (the side away from the source).

| Property | Distribution |
|---|---|
| Length | uniform, 2–20 cm |
| Width | log-uniform, 0.05–1 cm |
| Depth | uniform, 0.5–5 cm (5 cm = through-wall) |
| Orientation | uniform, 0–180° |
| Centre | uniform within ±(20 − length/2) cm |

**Dataset.** There are 600 walls, numbered `run_00000` to `run_00600` with `run_00076` missing, and 287 of them are cracked. They're split 70/15/15, stratified on cracked/clean with a fixed seed:

| Split | Walls | Cracked | Clean |
|---|---|---|---|
| Train | 420 | 201 | 219 |
| Validation | 90 | 43 | 47 |
| Test | 90 | 43 | 47 |

## 2. Method

**Inputs.** Each of the three maps is divided by the average clean training wall, logged, clamped to ±1 and standardised per channel. All of these statistics come from training walls only. A crack shows up as a small local excess of flux. In the peak channel it isn't visible by eye.

**Labels.** Each crack is drawn as a rotated rectangle on the 64 × 64 grid, giving a crack mask. Clean walls get an empty mask. The point source magnifies the crack's shadow on its way to the detector, so the drawing uses a fixed magnification of 1.25. That makes one pixel 0.5 cm at the shield, so the image covers about ±16 cm of the wall. Cracks narrower than 2 pixels are drawn 2 pixels wide.

**Network.** A 3-level U-Net:
- **Encoder:** 8, 16 and 32 channels, then a 64-channel bottleneck at 8 × 8. Each level is two 3 × 3 convolutions with batch norm and ReLU.
- **Decoder:** transposed-convolution upsampling with skip connections from the encoder.
- **Output:** a 1 × 1 convolution giving one logit per pixel.

**Training.**
- **Loss:** binary cross-entropy on every wall, with crack pixels weighted ×50, plus Dice loss on cracked walls.
- **Optimiser:** Adam, learning rate 10⁻³, batch size 16, 150 epochs. The epoch with the lowest validation loss is kept.
- **Augmentation:** random horizontal and vertical flips, transposes, and Gaussian noise (σ = 0.2).
- **Ensemble:** five models, identical except for the random seed.

**Measuring a crack.** The five probability maps are averaged. The crack is taken as the largest connected region above 0.5, and measured with probability-weighted image moments:
- **Centre:** the weighted mean position.
- **Orientation:** the principal axis of the weighted covariance.
- **Length:** √(12 × variance along that axis + 1), measured on the looser region above 0.25 so that the faint ends are kept.

Pixels are then converted to cm. Width (mostly smaller than one pixel) and depth are not measured.

**Detecting a crack.** The crack score is the highest value on the averaged probability map. A wall counts as cracked if its score is above a cut-off. The cut-off is chosen on the **clean validation walls** for a target false-alarm rate (1%, 5% or 10%), then applied unchanged to the test set.

## 3. Results

### 3.1 Measurement

Errors are mean absolute errors, measured on the cracked walls where any pixel is above 0.5. The baseline always guesses the training average.

| | Validation | Baseline | **Test** | Baseline |
|---|---|---|---|---|
| x | 1.93 cm | 7.28 | **1.20 cm** | 8.44 |
| y | 1.11 cm | 6.39 | **1.52 cm** | 7.04 |
| length | 1.29 cm | 4.10 | **1.85 cm** | 5.33 |
| angle | 5.6° | 44.0° | **10.8°** | 41.3° |
| cracks measured | 40/43 | | 34/43 | |

- **Length bias.** Predicted length is −0.07 cm too short on average on validation and +0.35 cm too long on test.
- **Ceiling.** Applying the same measurement method to the *true* masks gives a length error of 0.47 cm (validation) and 0.62 cm (test). That's the floor set by the method itself, even with a perfect network.

> **TODO:** add test MAE on cracks fully in view. Run `unet.py` with `EVAL_ON = 'test'` and copy the "cracked walls fully in view" block.

### 3.2 Detection

| | Validation | **Test** |
|---|---|---|
| AUC | 0.941 | **0.819** |
| Score > 0.5 | 40/43 found, **21/47 false alarms (45%)** | 34/43 found, **21/47 false alarms (45%)** |
| Cut-off 0.960 (1% target) | 34/43 (79%), 0/47 false alarms | **27/43 (63%), 1/47 (2%)** |
| Cut-off 0.913 (5% target) | 36/43 (84%), 2/47 (4%) | **27/43 (63%), 1/47 (2%)** |
| Cut-off 0.816 (10% target) | 38/43 (88%), 4/47 (9%) | **30/43 (70%), 4/47 (9%)** |

- **The 0.5 cut-off is unusable for detection.** It flags almost half of all clean walls.
- **Cut-offs chosen on validation hold up on test.** False-alarm rates came out close to their targets: 2% and 9%.
- **Detection is weaker on test than on validation.** Part of that gap comes from cracks that are out of view (section 4.1).

> **TODO:** add detection on cracks at least half in view. Copy the "in view x/y" figures from the same run.

### 3.3 Development history

**More data.** Test-set results as the dataset grew:

| Walls | x | y | length | angle | AUC | found |
|---|---|---|---|---|---|---|
| ~250 | 2.24 cm | 2.53 cm | 2.93 cm | 22.1° | 0.781 | 17/19 |
| ~400 | 2.02 cm | 1.77 cm | 1.87 cm | 17.8° | 0.854 | 25/29 |
| 600 | 1.20 cm | 1.52 cm | 1.85 cm | 10.8° | 0.819 | 34/43 |

Every increase helped measurement. Each row has a different test split, so the AUC changes are within split-to-split noise and shouldn't be read as a trend.

**Kept:** the uncollided flux as a third input channel, and the five-seed ensemble.

**Rejected:**

- **A crack/no-crack head on the bottleneck, trained alongside the masks.** On validation it gave no detection gain (AUC 0.953 against 0.941; 41/43 against 40/43 found). It made every measurement worse, for example angle 7.8° against 5.6°. The likely cause is that its noisy loss made training stop at an earlier epoch.
- **A depth-regression head.** It never did better than guessing the average depth, so it was dropped.

**Earlier approach.** A whole-image CNN classifier, scored with 5-fold cross-validation on about 250 walls, reached AUC 0.827. A simple brightest-pixel rule reached 0.832, so the two were tied. The classifier found 60.5% of cracks at a 0.8% false-alarm rate. The U-Net detects at a similar rate. Its added value is that it measures the crack as well as finding it.

### 3.4 Newly generated walls

Three new walls were simulated with seeds not used for any training run. All three were clean:

- **Two were correctly rejected:** crack scores 0.11 and 0.24.
- **One was a confident false alarm:** score 0.99, all five models agreeing, with a tight spread in position (±0.4 cm).

Three walls is too few to estimate a rate. The false alarm does show that the ensemble's spread can't be relied on to flag a wrong answer.

Two earlier "new" walls were excluded. They had been simulated with the same random seed as training wall `run_00001`, so their noise was largely a copy of that wall's.

## 4. Limitations and flaws

### 4.1 Cracks outside the detector's field of view

The generator keeps each crack within ±20 cm of the centre *at the shield*. But the shadow is magnified on its way to the detector, so the 40 cm detector only sees about ±16 cm of the shield. For the front of the deepest cracks it's about ±14.3 cm, because that part is magnified by 38.5 / 27.5 ≈ 1.4. Some cracks are therefore cut off, or lie entirely off the image. They're still labelled as cracked.

Cracked walls with fewer than 10 crack pixels in view:

| Split | Wholly out of view | Fewer than 10 pixels in view |
|---|---|---|
| Train | 8 | 18 |
| Validation | 0 | 1 |
| Test | 3 | 5 |

**Effects:**

- **Unfindable cracks count as misses.** About 12% of the test cracks can't be seen, but are still counted as misses. Validation has almost none, which explains part of the gap between validation and test (AUC 0.941 against 0.819).
- **Training is barely affected.** A crack that's wholly out of view has an empty mask, so for training it's simply another clean wall.

The "fewer than 10 pixels" count is only approximate, because a fully visible 2 cm crack covers about 6 pixels. The final evaluation script measures the *fraction* of each crack that's in view instead.

**Fix (not applied):** keep crack centres within ±(14.3 − length/2 − width/2) cm. A corrected generator is written but wasn't used to regenerate the data.

### 4.2 Inflated probabilities and false alarms

Training weights crack pixels ×50, so it pushes every probability upwards. A pixel showing 0.5 is nowhere near a 50% chance of crack. On top of that, the crack score is the highest of 4096 pixels, so a single crack-like noise streak anywhere is enough to raise it. Together these cause the 45% false-alarm rate at 0.5.

Choosing the cut-off for a set false-alarm rate fixes the reporting, but not the underlying behaviour. Possible fixes are a lower pixel weight, or a crack score based on the size of the detected region rather than its single highest pixel.

### 4.3 Small evaluation sets

Validation and test each contain only 43 cracked and 47 clean walls:

- **AUC is uncertain** by roughly ±0.04 at this size.
- **The 1% false-alarm cut-off rests on a single wall:** it's simply the highest-scoring clean validation wall.
- **Results move between splits.** The same models score AUC 0.941 on validation and 0.819 on test.

The U-Net wasn't cross-validated. The data-growth comparison in 3.3 also uses a different test split for each row.

### 4.4 Approximate magnification in the labels

Masks are drawn with one fixed magnification (1.25). The true value depends on how deep in the wall each part of the crack sits: from 38/32.5 ≈ 1.17 at the back face to 38/27.5 ≈ 1.38 at the front. So masks for deep cracks are slightly misplaced and undersized. This adds to the position and length errors, and partly explains why the length ceiling isn't zero.

### 4.5 Parameters not measured

Depth isn't measured: a dedicated network head never beat guessing the average. Width isn't measured either, because most cracks are narrower than one pixel (0.5 cm at the shield).

### 4.6 Idealised physics and measurement

- **The uncollided channel isn't measurable in practice.** It needs an energy window of ±100 eV around 662 keV, far narrower than any real detector's energy resolution. A real system couldn't separate uncollided photons from small-angle scatter, so this input channel isn't physically available. Detector efficiency and response are also ideal.
- **The detector is idealised.** It's a perfect 64 × 64 flux map in one exposure, not a scanned detector with a finite size.
- **There is only one geometry:** one wall thickness, one source position, one set of distances, and no background radiation.
- **The cracks are idealised:** straight, air-filled rectangular slots, always open on the back face. There's no reinforcing steel, no voids and no variation in concrete density.

### 4.7 Data-handling issues found during testing

- **Seed reuse.** Reusing a training run's seed makes a "new" wall largely a copy of that training wall, because OpenMC follows the same particle paths wherever the geometry matches. Walls made this way were excluded (3.4).
- **Overwrite risk.** The generator's default starting run number (501) would overwrite existing training files if run again. Neither issue affected the dataset used here: the test results reproduced exactly when re-run.

## 5. Future work

1. **Regenerate the data with every crack in view** (corrected generator already written), and report detection by how much of each crack is visible.
2. **Lower the crack-pixel weight**, or score walls on the size of the detected region, to cut false alarms. Compare versions by cracks found at a 5% false-alarm rate rather than by AUC.
3. **Add more walls, or more photons per wall.** Noise is the main limit, and every increase in data has helped so far.
4. **Use k-fold cross-validation** and larger, separately generated test sets for tighter error bars.
5. **Plot probability of detection against crack depth and width**, to separate model failures from cracks that are physically invisible.
6. **Use a realistic detector response** (energy resolution, efficiency, scanning), dropping the uncollided channel or replacing it with something measurable.
7. **Estimate depth** from the strength of the crack's signal in the transmitted flux, instead of a learned head.

## 6. Reproducing the results

| File | Purpose |
|---|---|
| Data generator (original version) | Simulates the walls into `data/run_XXXXX.npz` |
| `unet.py` | Trains or loads the ensemble, prints all results above, writes `crack_params_unet.csv` |
| `unet_unc_seed0.pt` – `unet_unc_seed4.pt` | The five trained models |
| `DrawCrack.py` | Plots one wall's prediction, uncertainty and true crack |
| `plot_check.py` | Plots one wall's flux maps (and its geometry, if its run folder exists) |

To reproduce:

1. In `unet.py`, set `TRAIN = False`.
2. Run it with `EVAL_ON = 'val'` for the validation results.
3. Run it again with `EVAL_ON = 'test'` for the test results.
