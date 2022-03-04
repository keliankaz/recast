from pathlib import Path
from typing import Any, Dict, Union

import torch

default_catalogs_dir = Path(__file__).parents[2] / "data"


class Catalog:
    """Earthquake catalog that consists of multiple datasets.

    Args:
        root_dir: Directory where all files of the dataset should be stored.
        metadata: Information about the catalog.
    """

    def __init__(self, root_dir: Union[str, Path], metadata: Dict[str, Any]):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.metadata = metadata
        if self.root_dir.exists():
            existing_metadata = torch.load(self.root_dir / "metadata.pt")
            # Check if the saved dataset is what we requested
            if existing_metadata == self.metadata and self.all_paths_exist():
                print(f"Loading existing catalog from {self.root_dir}.")
            else:
                raise FileExistsError(
                    f"A different catalog already exists in {self.root_dir}. "
                    f"There are two ways to fix this problem:\n"
                    f"  - Specify a different location as `root_dir`.\n"
                    f"  - Remove the existing catalog with\n     rm -rf {self.root_dir}"
                )
        else:
            print("Generating the catalog...")
            self.root_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.metadata, self.root_dir / "metadata.pt")
            self.generate_catalog()
            print(f"Catalog saved to {self.root_dir}")
            if not self.all_paths_exist():
                missing_paths = "\n  - ".join(
                    str(path.name) for path in self.required_paths if not path.exists()
                )
                raise RuntimeError(
                    "Catalog generation finished, but following required files haven't "
                    "been generated:\n  - "
                    f"{missing_paths}"
                    "\nOne of the methods `generate_catalog` or `required_files` "
                    "isn't implemented correctly."
                )

    def generate_catalog(self):
        """Create the dataset from scratch and save it to self.root_dir."""
        raise NotImplementedError

    @property
    def required_files(self):
        """Names of files that the dataset consists of."""
        raise NotImplementedError

    @property
    def required_paths(self):
        """Full paths to the files that the dataset consists of."""
        return [self.root_dir / file for file in self.required_files]

    def all_paths_exist(self):
        """Check if all required files exist."""
        return all([path.exists() for path in self.required_paths])

    def __repr__(self):
        return f"{self.__class__.__name__}(root_dir={self.root_dir})"
