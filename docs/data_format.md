# Data Format

## Trial list

Each line should contain:

```text
<label> <enroll_wav> <test_wav>
```

`label=1` denotes target/same-speaker trials; `label=0` denotes non-target/different-speaker trials.

## Gender map

Speaker-level JSON:

```json
{
  "id10270": "m",
  "id10300": "f"
}
```

## Cohort list for AS-Norm

One wav path per line. Absolute paths are recommended.
