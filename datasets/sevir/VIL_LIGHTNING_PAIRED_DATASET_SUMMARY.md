# Paired SEVIR VIL + Lightning Dataset Summary

This note summarizes the paired VIL-lightning dataset stored in:

- `datasets/sevir/data/paired_vil_lght_event/nowcast_training_vil.h5`
- `datasets/sevir/data/paired_vil_lght_event/nowcast_training_lght.h5`
- `datasets/sevir/data/paired_vil_lght_event/nowcast_training_META.csv`
- `datasets/sevir/data/paired_vil_lght_event/nowcast_validation_vil.h5`
- `datasets/sevir/data/paired_vil_lght_event/nowcast_validation_lght.h5`
- `datasets/sevir/data/paired_vil_lght_event/nowcast_validation_META.csv`
- `datasets/sevir/data/paired_vil_lght_event/nowcast_testing_vil.h5`
- `datasets/sevir/data/paired_vil_lght_event/nowcast_testing_lght.h5`
- `datasets/sevir/data/paired_vil_lght_event/nowcast_testing_META.csv`

The goal here is to document:

- how many train / validation / test items there are
- how those event-level cubes expand into training samples
- whether train / validation / test appear distribution-shifted

## 1. Dataset Construction

The paired dataset is produced by:

- `datasets/sevir/sevir_lightning_preprocessing.py`

Important preprocessing settings from:

- `datasets/sevir/data/paired_vil_lght_event/preprocessing_complete.json`

Settings:

- matched total paired events: `13,722`
- validation cutoff date: `2019-01-01 00:00:00`
- testing cutoff date: `2019-06-01 00:00:00`
- max `pct_missing`: `0.05`
- bbox tolerance: `0.001`
- no spatial downsampling (`downsample_factor = 1`)

Split rule:

- training: `time_utc < 2019-01-01`
- validation: `2019-01-01 <= time_utc < 2019-06-01`
- testing: `time_utc >= 2019-06-01`

## 2. Stored Event-Level Shapes

Each split stores paired event cubes:

| Split | VIL HDF5 shape | VIL dtype | Lightning HDF5 shape | Lightning dtype |
| --- | --- | --- | --- | --- |
| training | `(7593, 384, 384, 49)` | `uint8` | `(7593, 48, 48, 49)` | `int16` |
| validation | `(2737, 384, 384, 49)` | `uint8` | `(2737, 48, 48, 49)` | `int16` |
| testing | `(3392, 384, 384, 49)` | `uint8` | `(3392, 48, 48, 49)` | `int16` |

Interpretation:

- each metadata row corresponds to one paired event cube
- VIL is stored at `384 x 384 x 49`
- lightning is stored at native `48 x 48 x 49`

## 3. Training Sample Expansion

The lightning diffusion dataset uses:

- `raw_seq_len = 49`
- `lag_time = 13`
- `lead_time = 12`
- `stride = 12`

So each event yields:

- `1 + (49 - (13 + 12)) / 12 = 3` sliding-window samples

Therefore the effective sample counts used by the dataloader are:

| Split | Event cubes | Samples per event | Effective samples |
| --- | ---: | ---: | ---: |
| training | 7,593 | 3 | 22,779 |
| validation | 2,737 | 3 | 8,211 |
| testing | 3,392 | 3 | 10,176 |

## 4. Time Coverage

Event time coverage:

| Split | Time min | Time max |
| --- | --- | --- |
| training | `2018-02-03 06:32:30` | `2018-12-28 20:51:15` |
| validation | `2019-01-05 00:10:00` | `2019-05-29 21:00:00` |
| testing | `2019-06-11 16:05:00` | `2019-11-30 18:46:15` |

Monthly event counts:

### Training

- `2018-02`: 691
- `2018-03`: 460
- `2018-04`: 589
- `2018-05`: 362
- `2018-06`: 714
- `2018-07`: 1,275
- `2018-08`: 1,050
- `2018-09`: 597
- `2018-10`: 608
- `2018-11`: 674
- `2018-12`: 573

### Validation

- `2019-01`: 502
- `2019-02`: 467
- `2019-03`: 696
- `2019-04`: 550
- `2019-05`: 522

### Testing

- `2019-06`: 202
- `2019-07`: 466
- `2019-08`: 834
- `2019-09`: 694
- `2019-10`: 388
- `2019-11`: 808

Observation:

- this is a pure time split
- training is all from 2018
- validation is early 2019
- testing is mid/late 2019

So there is definite temporal distribution shift by construction.

## 5. Event Composition

SEVIR ids beginning with:

- `R` are random events
- `S` are storm-event-style catalog entries

Split composition by id prefix:

| Split | `R` count | `S` count | `%R` |
| --- | ---: | ---: | ---: |
| training | 6,043 | 1,550 | 79.6% |
| validation | 2,370 | 367 | 86.6% |
| testing | 2,871 | 521 | 84.6% |

Observation:

- all three splits are dominated by random events
- validation is the most random-event-heavy
- train/test have a somewhat larger fraction of storm-event cases than validation

Event type counts are also heavily dominated by missing labels because most `R` events do not carry severe-weather event labels in the same way as storm-event rows.

Top labeled event types:

### Training

- `Thunderstorm Wind`: 699
- `Hail`: 383
- `Flash Flood`: 177
- `Flood`: 85
- `Tornado`: 78

### Validation

- `Thunderstorm Wind`: 142
- `Hail`: 116
- `Flood`: 38
- `Tornado`: 33
- `Flash Flood`: 18

### Testing

- `Thunderstorm Wind`: 281
- `Hail`: 127
- `Flash Flood`: 49
- `Flood`: 17
- `Heavy Rain`: 16

## 6. VIL Distribution Signals

Using metadata `data_max` as a quick VIL intensity proxy:

| Split | Mean `data_max` | P50 | P90 |
| --- | ---: | ---: | ---: |
| training | 214.70 | 235.0 | 254.0 |
| validation | 196.31 | 203.0 | 254.0 |
| testing | 214.97 | 236.0 | 254.0 |

Observation:

- validation has noticeably lower mean / median VIL maxima than training and testing
- training and testing look much closer to each other in this radar-intensity proxy

This suggests validation may be a somewhat easier or weaker-storm regime than testing, at least on average intensity.

## 7. Lightning Distribution Signals

For model training and evaluation, the sample-window view is the most important one.
The model is trained on sliding windows, not on whole 49-frame event cubes directly.

Each event cube is expanded into 3 effective samples with:

- `13` past frames
- `12` future target frames

So Section 7.2 is the primary training-time distribution view. Section 7.1 is still
useful as an event-level reference, but it is not exactly the distribution the model
sees during optimization.

### 7.1 Event-level lightning totals

Total lightning counts per 49-frame event cube:

For one paired lightning event cube with shape `(48, 48, 49)`, define:

`event_total_count = sum(lightning_cube[y, x, t])` over all `48 x 48 x 49` cells

So in the table below:

- `Mean total count` = mean of `event_total_count` across all event cubes in that split
- `P50` = 50th percentile of `event_total_count` across event cubes in that split
- `P90` = 90th percentile of `event_total_count` across event cubes in that split

These are event-level totals over the full 49-frame cube, not per-frame totals.

| Split | Mean total count | P50 | P90 | Event has any lightning |
| --- | ---: | ---: | ---: | ---: |
| training | 2861.35 | 68.0 | 9384.4 | 61.5% |
| validation | 3030.17 | 2.0 | 9208.8 | 53.0% |
| testing | 4091.29 | 167.5 | 13947.7 | 67.4% |

Observation:
- testing is clearly the most lightning-active split
- validation is the sparsest split by median event-level lightning
- training sits in between

### 7.2 Sample-window lightning activity

These are the more relevant statistics for training, since the model sees 25-frame windows rather than whole 49-frame cubes.

For one effective sample:

- `sample_future_total_count = sum(future_lightning_window[y, x, t])` over all `48 x 48 x 12` future cells
- `Future window has lightning` means that at least one future cell in the `48 x 48 x 12` target window is positive
- `Future total P50` and `Future total P90` are percentiles of `sample_future_total_count` across effective samples

So this section directly reflects the windowed target distribution seen by the model.

Past/future lightning presence combinations over effective samples:

- `Past no, Future no`: no lightning anywhere in the past `48 x 48 x 13` window, and no lightning anywhere in the future `48 x 48 x 12` window
- `Past yes, Future no`: lightning exists in the past window, but not in the future target window
- `Past no, Future yes`: no lightning in the past window, but lightning appears in the future target window
- `Past yes, Future yes`: lightning exists in both past and future windows

| Split | Past no, Future no | Past yes, Future no | Past no, Future yes | Past yes, Future yes |
| --- | ---: | ---: | ---: | ---: |
| training | 42.2% (`9,617`) | 3.5% (`789`) | 2.6% (`598`) | 51.7% (`11,775`) |
| validation | 51.9% (`4,263`) | 4.6% (`378`) | 3.3% (`275`) | 40.1% (`3,295`) |
| testing | 36.7% (`3,731`) | 3.6% (`370`) | 2.9% (`299`) | 56.8% (`5,776`) |

Observation:

- the dominant state in all splits is `Past yes, Future yes`, especially in testing
- validation has the largest `Past no, Future no` share, so it is the sparsest split at the sample-window level
- `Past no, Future yes` is relatively rare in all splits, meaning truly new lightning onset without any past-window lightning is uncommon
- `Past yes, Future no` is also relatively rare, meaning once lightning is present in the past window it often persists into the future target window

VIL heaviness proxy inside each transition class:

To keep this summary lightweight, the table below uses the event metadata field
`data_max` as a proxy for rainfall / VIL heaviness, repeated across the 3 sliding-window
samples generated from the same event cube. So this is a coarse event-level VIL proxy,
not an exact sample-window past-VIL statistic.

Bins:

- `H`: `data_max >= 133`
- `M`: `74 <= data_max < 133`
- `L`: `16 <= data_max < 74`
- `<16`: `data_max < 16`

Percentages below are within each transition column.

| Split | Past no, Future no | Past yes, Future no | Past no, Future yes | Past yes, Future yes |
| --- | --- | --- | --- | --- |
| training | `42.2% (9,617)`<br>`H/M/L/<16 = 81.4 / 13.4 / 4.5 / 0.6` | `3.5% (789)`<br>`H/M/L/<16 = 98.7 / 1.3 / 0.0 / 0.0` | `2.6% (598)`<br>`H/M/L/<16 = 98.2 / 1.7 / 0.2 / 0.0` | `51.7% (11,775)`<br>`H/M/L/<16 = 100.0 / 0.0 / 0.0 / 0.0` |
| validation | `51.9% (4,263)`<br>`H/M/L/<16 = 73.2 / 20.1 / 5.5 / 1.1` | `4.6% (378)`<br>`H/M/L/<16 = 98.1 / 1.9 / 0.0 / 0.0` | `3.3% (275)`<br>`H/M/L/<16 = 98.5 / 1.1 / 0.4 / 0.0` | `40.1% (3,295)`<br>`H/M/L/<16 = 100.0 / 0.0 / 0.0 / 0.0` |
| testing | `36.7% (3,731)`<br>`H/M/L/<16 = 77.2 / 16.8 / 5.1 / 0.9` | `3.6% (370)`<br>`H/M/L/<16 = 98.6 / 1.4 / 0.0 / 0.0` | `2.9% (299)`<br>`H/M/L/<16 = 99.3 / 0.7 / 0.0 / 0.0` | `56.8% (5,776)`<br>`H/M/L/<16 = 100.0 / 0.0 / 0.0 / 0.0` |

Hard-negative interpretation:

- most `Past yes, Future no` and `Past no, Future yes` samples are associated with very high event-level VIL maxima
- even `Past no, Future no` is often not a weak-rain case; a large fraction still belongs to the heavy-VIL proxy bin
- this supports the idea that many of the negative samples are genuine hard negatives, not only trivial dry-background cases

Future window statistics over all effective samples:

| Split | Effective samples | Future window has lightning | Mean future occupancy fraction | Mean future total count | Future total P50 | Future total P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 22,779 | 54.3% | 0.00764 | 688.62 | 5.0 | 2243.2 |
| validation | 8,211 | 43.5% | 0.00702 | 725.07 | 0.0 | 2184.0 |
| testing | 10,176 | 59.7% | 0.00920 | 974.33 | 19.0 | 3174.0 |

Equivalent future no-lightning sample fractions:

| Split | Future window has no lightning |
| --- | ---: |
| training | 45.7% |
| validation | 56.5% |
| testing | 40.3% |

Conditional future-window intensity, only over samples where the future window has any lightning:

| Split | Future-lightning samples | Mean future total count | Future total P50 | Future total P90 |
| --- | ---: | ---: | ---: | ---: |
| training | 12,373 | 1267.77 | 462.0 | 3636.8 |
| validation | 3,570 | 1667.67 | 302.0 | 5047.4 |
| testing | 6,075 | 1632.06 | 447.0 | 4936.6 |

Past window statistics over all effective samples:

| Split | Past window has lightning | Mean past occupancy fraction | Mean past total count |
| --- | ---: | ---: | ---: |
| training | 55.2% | 0.00801 | 794.41 |
| validation | 44.7% | 0.00733 | 845.21 |
| testing | 60.4% | 0.00976 | 1151.48 |

Conditional past-window intensity, only over samples where the past window has any lightning:

| Split | Past-lightning samples | Mean past total count | Past total P50 | Past total P90 |
| --- | ---: | ---: | ---: | ---: |
| training | 12,564 | 1440.29 | 535.0 | 4140.8 |
| validation | 3,673 | 1889.47 | 335.0 | 5784.8 |
| testing | 6,146 | 1906.51 | 538.5 | 5719.0 |

Observation:

- validation is again the sparsest split
- testing is the most active split
- testing has the highest future occupancy fraction and the highest future lightning totals
- this means a model tuned mainly on validation may under-estimate the denser lightning regime seen in test
- more than half of validation samples have no lightning in the future target window
- testing is materially less sparse than validation at the sample-window level

## 8. Geographic and Quality Checks

Quick consistency checks:

| Split | Mean center lat | Mean center lon | Mean `pct_missing` | Max `bbox_diff` |
| --- | ---: | ---: | ---: | ---: |
| training | 37.38 | -92.77 | 0.000248 | ~`1.28e-13` |
| validation | 37.29 | -93.90 | 0.000519 | ~`5.68e-14` |
| testing | 37.73 | -92.72 | 0.000286 | ~`1.42e-13` |

Interpretation:

- geographic centers are fairly similar across splits
- missingness is very low in all splits
- bbox matching is essentially exact numerically

So the strongest split differences are temporal / event-mixture / lightning-activity differences, not obvious georegistration problems.

## 9. Main Conclusion on Distribution Shift

Yes, there appears to be meaningful train / validation / test distribution shift.

The clearest signals are:

- time shift:
  training is 2018, validation is early 2019, testing is later 2019
- composition shift:
  validation is more dominated by random-event cases than training
- radar-intensity shift:
  validation has lower mean / median `data_max`
- lightning-activity shift:
  validation is sparsest, testing is most active

The most important practical takeaway is:

- at the sample-window level, validation is the sparsest split
- at the sample-window level, testing is the most lightning-active split

So a checkpoint selected only on validation loss may not perfectly represent test-time behavior, especially if the model under-predicts denser lightning windows.

## 10. Recommended Next Checks

If we want to study the split mismatch more carefully, the next useful diagnostics would be:

- exact train / val / test distributions of future positive-pixel rate at the sample level
- buffered / pooled CSI by split, since pixel-exact lightning is spatially ambiguous
- month-by-month lightning activity curves
- calibration comparison between validation and testing
- train / val / test histograms of mean raw forecast intensity and chosen CSI threshold
