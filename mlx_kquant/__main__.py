"""Enable ``python -m mlx_kquant`` as an alias for the ``mlx-kquant`` CLI."""

from .cli import main

raise SystemExit(main())
