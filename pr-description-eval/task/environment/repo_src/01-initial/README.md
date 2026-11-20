# logslice

`logslice` filters and summarises structured log files from the command line.
It understands lines of the form

```
2024-05-01T12:00:00Z INFO [http] GET /index.html 200
```

(ISO-8601 timestamp, level, `[component]`, message). Lines that don't match are
skipped. `WARN`, `ERR` and `FATAL` are accepted as aliases for `WARNING`,
`ERROR` and `CRITICAL`.

## Install

```
pip install -e .
```

## Usage

```
logslice app.log --level warning             # WARNING and above
logslice app.log -c db -c http               # only these components
logslice app.log --since 2024-05-01T12:00:00Z --until 2024-05-01T13:00:00Z
logslice app.log --grep 'timeout|reset'      # regex on the message
logslice app.log --json                      # JSON lines output
cat app.log | logslice -                     # read from stdin
```

## Development

```
pip install -e . pytest
pytest
```
