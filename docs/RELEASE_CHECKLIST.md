# Journal release checklist

This checklist records work that should be completed before the repository is made public and archived.

- [x] Replace legacy sparse-autoencoder README.
- [x] Document final datasets, subject inclusion, and event mappings.
- [x] Identify final journal experiment configuration in `oddball_eeg.config`.
- [x] Preserve full development environment and add a portable environment.
- [x] Document notebook roles and the regeneration workflow.
- [x] Add software license and citation metadata.
- [x] Package reusable code (`src/oddball_eeg`), organise notebooks, strip outputs.
- [x] Add unit/smoke tests and continuous integration.
- [ ] Verify the portable environment from a clean checkout.
- [ ] Verify the exact OpenNeuro snapshot/version identifiers and dataset citations.
- [ ] Re-run at least dataset loading and one representative analysis path on the real data after the package refactor (e.g. notebook 02 for ds003061) and compare against the thesis tables.
- [ ] Regenerate any lightweight derived result files intended for public distribution.
- [ ] Confirm CRediT/authorship and public-release approval with all authors.
- [ ] Create a versioned GitHub release.
- [ ] Archive the release with Zenodo and add the DOI to `CITATION.cff`, README, manuscript, and Supplementary Material.
