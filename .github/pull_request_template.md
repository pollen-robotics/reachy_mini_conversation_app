## Summary
<!-- What does this PR change and why? -->

## Category
- [ ] Fix
- [ ] Feature
- [ ] Refactor
- [ ] Docs
- [ ] CI/CD
- [ ] Other

## Check before merging
### Basic
- [ ] CI green (Ruff, Tests, Mypy)
- [ ] Code update is clear (types, docs, comments)

### Run modes
- [ ] Headless mode (default)
- [ ] Gradio UI (`--gradio`)
- [ ] Everything is tested in simulation as well (`--gradio` required)

### Vision / motion
- [ ] Local vision (`--local-vision`)
- [ ] YOLO or MediaPipe head tracker (`--head-tracker {yolo,mediapipe}`)
- [ ] Camera pipeline (with/without `--no-camera`)
- [ ] Movement manager (dances, emotions, head motion)
- [ ] Head wobble
- [ ] Profiles or custom tools

### Dependencies & config
- [ ] Updated `.env.example` if new config vars added

## Notes
<!-- Optional: context, caveats, migration notes -->
