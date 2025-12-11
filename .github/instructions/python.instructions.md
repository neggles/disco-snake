---
description: "Python coding conventions and guidelines"
applyTo: "**/*.py"
---

# Python Coding Conventions

## Python Instructions

-   Check pyproject.toml for the minimum supported Python version.
-   Write clear and concise comments for each function.
-   Ensure functions have descriptive names and include type hints.
-   Provide docstrings following PEP 257 conventions.
-   Do _not_ use the `typing` module for stdlib type annotations when unnecessary (e.g., use `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`).
-   Break down complex functions into smaller, more manageable functions.

## General Instructions

-   Always prioritize readability and clarity.
-   For algorithm-related code, include explanations of the approach used.
-   Write code with good maintainability practices, including comments on why certain design decisions were made.
-   Handle edge cases and write clear exception handling.
-   For libraries or external dependencies, mention their usage and purpose in comments.
-   Use consistent naming conventions and follow language-specific best practices.
-   Write concise, efficient, and idiomatic code that is also easily understandable.

## Code Style and Formatting

-   Follow the **PEP 8** style guide for Python.
-   Maintain proper indentation (use 4 spaces for each level of indentation).
-   Follow `ruff` settings in `pyproject.toml` for linting and formatting.
-   Place function and class docstrings immediately after the `def` or `class` keyword.
-   Use blank lines to separate functions, classes, and code blocks where appropriate.
-   Run `ruff check` and `ruff format` before committing code to ensure compliance with style guidelines.
-   Run `pre-commit` hooks (if configured) to automatically format code and catch common issues.

## Edge Cases and Testing

-   Always include test cases for critical paths of the application.
-   Account for common edge cases like empty inputs, invalid data types, and large datasets.
-   Include comments for edge cases and the expected behavior in those cases.
-   Write unit tests for functions and document them with docstrings explaining the test cases.

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.

    Parameters:
    radius (float): The radius of the circle.

    Returns:
    float: The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```
