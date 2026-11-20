# Contributing to logslice

## Setup

```
pip install -e . pytest
pytest
```

## Pull requests

- Keep each PR focused on one change.
- Add or update tests for every behaviour change; don't delete or weaken
  existing tests to make them pass.
- Update `README.md` when you change user-facing flags or behaviour.
- Keep the PR description in sync with the code when you push changes:
  reviewers and release notes rely on it, not just the diff.
- Address review comments with follow-up commits (no force-pushes) so
  reviewers can see what changed.
