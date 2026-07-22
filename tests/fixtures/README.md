# Test fixtures

Drop a **classroom / person image** here (`.jpg`, `.jpeg`, or `.png`) to make
`test_detects_person_on_fixture` run against a representative frame.

If this directory has no image, the test falls back to Ultralytics' bundled
sample images (`bus.jpg`, `zidane.jpg`) — both contain people — so the test
still runs for real on a machine with `ultralytics` installed. It only skips if
neither a local fixture nor the Ultralytics assets are available.
