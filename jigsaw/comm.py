"""Functions relating to process groups."""


def shutdown_comm(process_group_dict: dict) -> None:
    """Shutdown existing process groups and mesh."""
    for _, pg in process_group_dict.items():
        pg.shutdown()
