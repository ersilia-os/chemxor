"""Hello world nodes."""

from kedro.pipeline import node


# Prepare first node
def return_greeting() -> str:
    """Greetings node.

    Returns:
        str: [description]
    """
    return "Hello"


return_greeting_node = node(func=return_greeting, inputs=None, outputs="my_salutation")


# Prepare second node
def join_statements(greeting: str) -> str:
    """Joining node.

    Args:
        greeting (str): Greetings

    Returns:
        str: [description]
    """
    return f"{greeting} Kedro!"


join_statements_node = node(
    join_statements, inputs="my_salutation", outputs="my_message"
)
