# Dataset configuration

The journal study uses four public OpenNeuro EEG datasets.

| Dataset | Final analyzed condition | Included subjects |
|---|---|---:|
| `ds003061` | Auditory oddball / P300 | 13 |
| `ds005863` | Visual oddball | 30 |
| `ds006466` | Active auditory oddball | 30 |
| `ds006480` | Pooled auditory oddball | 30 |

Total: **103 subject-recordings**.

## Expected local layout

The recommended layout is shown below. Alternatively set `ODDBALL_EEG_DATA_DIR` to the folder containing the dataset directories; `oddball_eeg.bids.bids_root_candidates()` lists every location searched.

```text
datasets/
└── openneuro/
    ├── ds003061/
    ├── ds005863/
    ├── ds006466/
    └── ds006480/
```

Each dataset directory should contain its BIDS `dataset_description.json`.

## Subject inclusion

### ds003061

Subjects `001` through `013`.

### ds005863

Subjects `001` through `030`.

### ds006480

Subjects `01` through `30`.

### ds006466

The fixed 30-subject set used in the final analysis is:

`AgoBurn`, `AgoHake`, `ArcoWeb`, `AridWar`, `AuldEve`, `AvidJar`, `AzoAgon`, `BoneSip`, `BossBob`, `BossRip`, `CalmJib`, `CampHum`, `CampOhm`, `DearSop`, `DrySolo`, `EpicAwl`, `EpicOla`, `EvenBox`, `EyedEgo`, `FarTube`, `FeyHeat`, `FirmPhi`, `FitHame`, `FivePly`, `FourSpy`, `HeldSou`, `HepBleu`, `HepViol`, `HewnMod`, and `HomeDye`.

## Event mappings

All datasets are converted to binary labels: standard = 0 and target = 1.

### ds003061

- Standard: `stimulus/standard`
- Targets: `stimulus/oddball`, `stimulus/oddball_with_response`
- The loader also accepts the historical source annotation `stimulus/oddball_with_reponse` when present.

### ds005863

Targets:

`11`, `22`, `33`, `44`, `55`

Included standards:

`12`, `13`, `14`, `15`, `21`, `23`, `24`, `25`, `31`, `32`, `34`, `35`, `41`, `42`, `43`, `45`, `51`, `52`, `53`, `54`

### ds006466

- Standard: `113`
- Target: `114`
- Final benchmark condition: active auditory oddball.

### ds006480

- Standards: `106`, `113`, `125`, `132`
- Targets: `107`, `114`, `126`, `133`
- Excluded distractors: `108`, `115`, `127`, `134`

The final benchmark pools the included auditory-oddball conditions.

## Data handling

Raw OpenNeuro data are not redistributed in this repository. Event extraction and raw preprocessing are implemented in `oddball_eeg.bids`, chronological trial reconstruction and chunk grouping in `oddball_eeg.loader`, and cropping/normalisation in `oddball_eeg.preprocessing`.

**On-disk header repair.** Some BrainVision recordings (ds005863) contain `.vhdr` headers whose `DataFile`/`MarkerFile` entries still point to pre-BIDS filenames. `oddball_eeg.bids.repair_brainvision_header_links` rewrites those two entries to the BIDS filenames before reading, keeping a one-time `.vhdr.bak` backup next to the original. No signal data are modified.

## OpenNeuro versions

The manuscript bibliography records the dataset versions used during the study. Before the public archival release, these versions and permanent identifiers should be checked once more against the corresponding OpenNeuro records. See `RELEASE_CHECKLIST.md`.
