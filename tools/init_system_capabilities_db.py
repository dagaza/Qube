from __future__ import annotations

from core.system_capabilities_store import SystemCapabilitiesStore, default_system_data_dir


def main() -> None:
    store = SystemCapabilitiesStore()
    print(f"Initialized system capabilities DB at: {store.db_path}")
    print(f"System data directory: {default_system_data_dir()}")
    print(f"Curated registry: {store.registry_path}")
    print(f"Learned registry: {store.learned_registry_path}")
    print(f"Missed models log: {store.missed_models_path}")


if __name__ == "__main__":
    main()
