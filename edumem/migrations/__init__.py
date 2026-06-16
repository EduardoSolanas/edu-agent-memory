"""
edumem schema migrations
===========================

Idempotent, package-shipped schema migrations. Lives under the
`edumem` package (rather than `scripts/`) so the migration logic is
present on every install path — pip wheels, editable installs, source
checkouts. CLI wrappers in `scripts/` import from here.
"""
