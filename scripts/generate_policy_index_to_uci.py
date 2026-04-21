from __future__ import annotations

import argparse
import json
from pathlib import Path

from maia_api.policy_map import validate_lc0_policy_index_to_uci, write_lc0_policy_index_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Lc0-compatible 1858 policy_index_to_uci JSON mapping."
    )
    parser.add_argument(
        "--output",
        default="models/policy_index_to_uci.json",
        help="Output JSON path (default: models/policy_index_to_uci.json)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing mapping file at --output without regenerating it",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.validate_only:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            mapping = payload
        elif isinstance(payload, dict):
            parsed = {int(k): v for k, v in payload.items()}
            if set(parsed.keys()) != set(range(len(parsed))):
                raise ValueError("Mapping dict indexes must be contiguous from 0")
            mapping = [parsed[idx] for idx in range(len(parsed))]
        else:
            raise ValueError("Mapping file must be a JSON list or dict")

        validate_lc0_policy_index_to_uci(mapping)
        print(f"Validated Lc0 policy mapping at {output_path}")
        return

    write_lc0_policy_index_json(output_path)
    print(f"Wrote Lc0 policy mapping to {output_path} (1858 entries)")


if __name__ == "__main__":
    main()
