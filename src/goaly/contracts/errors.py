"""Contract exceptions: the errors one layer raises for another to handle (spec §3.3)."""

__all__ = ["EnvironmentFailure"]


# The design names it a failure of the environment, not an error of its caller.
class EnvironmentFailure(Exception):  # noqa: N818
    """The environment itself failed, e.g. the database connection or the container died; it is
    never a property of the agent's command.

    Environments raise it, and only for that case: a statement or command that fails gives a
    result. The tool executor re-raises it rather than turning it into an error result for the
    model, a policy lets it propagate, and the runner records the task as
    TASK_ERROR_TYPE_ENVIRONMENT.
    """
