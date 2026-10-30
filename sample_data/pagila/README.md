# Pagila sample dataset

A DVD-rental business database (films, actors, customers, rentals, payments, stores). It's not built
into the app: load it like any other dataset.

```bash
python -m app.datasets.cli load sample_data/pagila/pagila.sql.gz --name Pagila
```

It is also the reference dataset for the evaluation suite, which expects it loaded with the slug
`pagila` (the default for `--name Pagila`).

| | |
|---|---|
| File | `pagila.sql.gz`: `pagila-schema.sql` + `pagila-data.sql`, concatenated and gzipped |
| Source | https://github.com/devrimgunduz/pagila, commit `ce6e114236bb828e5ec0dc835f65f1bf1b9179f5` |
| License | PostgreSQL License, see `LICENSE.txt` |
| Requires | PostgreSQL 18 (`uuidv7()`, virtual generated columns) |

When loaded with the default rules, these columns are hidden from the bot: `staff.password`,
`staff.picture`, `film.fulltext`, `film_embedding.embedding`.
