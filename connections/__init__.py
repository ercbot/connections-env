from .environment import ConnectionsEnv

import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    return ConnectionsEnv(**kwargs)
