"""Earth Engine sign-in that works the same in Colab and on a laptop."""

from __future__ import annotations

import ee


def init(project_id: str) -> None:
    """Authenticate (if needed) and initialise Earth Engine under ``project_id``.

    ``project_id`` is a Google Cloud project that is registered for Earth
    Engine (https://code.earthengine.google.com/register). In Colab the first
    call opens a Google sign-in prompt; later calls reuse the saved token.
    """
    if not project_id or project_id == "your-ee-project-id":
        raise ValueError(
            "Set PROJECT_ID to your own Earth Engine Cloud project ID first "
            "(see https://code.earthengine.google.com/register)."
        )
    try:
        ee.Initialize(project=project_id)
    except Exception:  # not signed in yet, or the token expired
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print(f"Earth Engine ready (project: {project_id})")
