"""
Defining the RetryConfig class and the RetryStrategy class.
"""

from typing import TypeVar, Callable, Optional
import logging
from functools import wraps
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = Field(
        default=5,
        description="Maximum number of retry attempts before giving up",
        ge=1,  # ensures value is greater than or equal to 1
    )

    should_retry: Optional[Callable[[Exception], bool]] = Field(
        default=None,
        description=(
            "Optional function that takes an exception and "
            "returns True if retry should be attempted"
        ),
    )

    error_message: str = Field(
        default="Operation failed",
        description="Message to log when operation fails"
    )

    @field_validator("should_retry", mode="before")
    @classmethod
    def set_default_should_retry(cls, v):
        """Set default retry behavior to retry on all exceptions if no function provided."""
        return v if v is not None else lambda _: True


class RetryStrategy(RetryConfig):
    """A class to handle retry logic with customizable behavior."""
    def execute(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute the given operation with retry logic.

        Args:
            operation: The function to execute
            *args, **kwargs: Arguments to pass to the operation

        Returns:
            The result of the operation

        Raises:
            ValueError: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                result = operation(*args, **kwargs)
                return result
            except Exception as e:
                last_exception = e
                if not self.should_retry(e):
                    break

                logger.warning(
                    "Function '%s' %s: %s. Attempt %d/%d",
                    operation.__name__,
                    self.error_message.lower(),
                    str(e),
                    attempt + 1,
                    self.max_attempts,
                )

        raise ValueError(
            f"Function '{operation.__name__}' {self.error_message.lower()} "
            f"after {self.max_attempts} attempts. Last error: {str(last_exception)}"
        )


def retry(
    max_attempts: int = 5,
    should_retry: Optional[Callable[[Exception], bool]] = None,
    error_message: str = "Operation failed",
) -> Callable:
    """
    A decorator that applies retry logic to a function.

    Args:
        max_attempts: Maximum number of retry attempts before giving up.
            Must be at least 1. Default is 5.
        should_retry: Optional function that takes an exception and returns
            True if a retry should be attempted, False otherwise.
            If not provided, all exceptions will trigger a retry.
            Example:
                def retry_on_connection_error(exc: Exception) -> bool:
                    return isinstance(exc, ConnectionError)
        error_message: Message to log on failure

    Returns:
        A decorated function with retry logic

    Example:
        @retry(max_attempts=3, error_message="Failed to fetch data")
        def fetch_data():
            # Your code here
            pass
    """
    strategy = RetryStrategy(
        max_attempts=max_attempts, should_retry=should_retry, error_message=error_message
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return strategy.execute(func, *args, **kwargs)

        return wrapper

    return decorator

