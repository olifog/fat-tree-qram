# Fat-tree QRAM

Implementation of quantum RAM architectures based on [arXiv:2502.06767](https://arxiv.org/pdf/2502.06767)

## Usage

### Run BB QRAM Demo

```bash
uv run bb
```

This demonstrates an 8-address QRAM querying multiple addresses with 100% accuracy.

### Run Tests

```bash
uv run pytest tests/ -v
```

### Linting

```bash
uv run ruff check
```

### Type Checking

```bash
uv run mypy src
```
