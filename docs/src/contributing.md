# Contributing

Thank you for your interest in contributing to **TopicJev**!

## Project context

TopicJev is developed and maintained by an academic research group in connection with ongoing
research in natural language processing and topic modeling. Because we are active researchers
rather than a dedicated full-time support team:

- **Support:** we provide maintenance, address bug reports, and review contributions on a
  best-effort basis as time and academic commitments permit.
- **Scope:** changes that align with reproducibility, robustness, efficiency, and scientific rigor
  are especially welcome.
- **Openness:** we are genuinely open to community feedback, bug reports, and pull requests from
  external contributors.

## Reporting bugs

Check the existing [issues](https://github.com/anatems1/TopicJev/issues) first. If the problem
is new, open an issue with:

- a clear and descriptive title;
- a minimal reproducible example (code snippet and data sample);
- your environment: Python, PyTorch and Transformers versions, OS, GPU or CPU;
- the full traceback or error log.

## Suggesting enhancements

Open an issue to discuss the idea before investing substantial time in code. This helps make sure
the proposal fits the project's research scope and architecture.

## Pull requests

1. **Fork and branch:** fork the repository and create a feature branch
   (`git checkout -b feature/my-feature`).
2. **Focus:** keep each pull request to a single fix or feature. Small, self-contained pull
   requests are much easier to review and merge.
3. **Open the PR** with a concise description of the change, and link related issues.

## Building the documentation

This site is built with [MkDocs](https://www.mkdocs.org) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). The Markdown sources are in
`docs/src`, the configuration is `mkdocs.yml` in the repository root, and the API reference is
generated from the docstrings with [mkdocstrings](https://mkdocstrings.github.io).

```bash
pip install --group docs        # pip 25.1+; or: pip install mkdocs-material "mkdocstrings[python]"
mkdocs serve                    # live preview at http://127.0.0.1:8000
mkdocs build --strict           # writes the site to docs/html
```

## Code of conduct

We value a collaborative, respectful, and constructive environment. Please keep discussions
professional, polite, and focused on improving the software and research.
