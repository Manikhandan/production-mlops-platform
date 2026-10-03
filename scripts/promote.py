from __future__ import annotations

import argparse

from mlops_platform.config import get_settings
from mlops_platform.registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a registered model version.")
    parser.add_argument("version")
    parser.add_argument("--stage", default="Production")
    args = parser.parse_args()
    settings = get_settings()
    record = ModelRegistry(settings.registry_dir).promote(args.version, args.stage)
    print({"version": record.version, "stage": record.stage})


if __name__ == "__main__":
    main()
