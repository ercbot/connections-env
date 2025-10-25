import verifiers as vf

from .environment import ConnectionsEnv


def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    return ConnectionsEnv(**kwargs)
