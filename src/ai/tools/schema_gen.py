"""OpenAI tool schema definition models and helpers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel


class FunctionBaseModel(BaseModel):
    """Base model for tool call schemas with common config."""

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )


class FunctionParameter(FunctionBaseModel):
    param_type: str = Field(
        ...,
        description="The data type of the parameter.",
        alias="type",
    )
    description: str = Field(
        ...,
        description="A description of the parameter.",
    )
    enum: list[str] | None = Field(
        None,
        description="An optional enumeration of valid string values for the parameter (if its type is 'string').",
    )
    items: dict | None = Field(
        None,
        description="If the parameter is an array, this defines the schema of the array items.",
    )


class FunctionParameters(RootModel[dict[str, FunctionParameter]]):
    """Model for function parameters schema."""

    pass


class FunctionSchema(FunctionBaseModel):
    """Schema for defining a function for AI model function calling."""

    tool_type: Literal["function"] = Field(
        "function", description="The type of tool, always 'function'.", alias="type"
    )
    name: str = Field(..., description="The name of the function.")
    description: str = Field(..., description="A description of what the function does.")
    parameters: FunctionParameters = Field(
        ...,
        description="The parameters schema for the function, following JSON Schema specifications.",
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )
