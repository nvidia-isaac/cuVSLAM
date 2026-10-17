# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""Expose opaque LMDB operations as four typed NeMo Agent Toolkit functions."""

import asyncio
import json
import threading
from pathlib import Path

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, ConfigDict, Field

from .store import MapStore

_store_lock = threading.RLock()


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DescribeInput(Input):
    map_name: str = Field(description="Existing map directory relative to the configured root")


class CloneInput(Input):
    source: str
    destination: str = Field(description="New directory below the root; must not exist")


class ReadInput(DescribeInput):
    table: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=500, ge=1, le=5000)


class Change(Input):
    table: str
    key_hex: str
    expected_sha256: str | None = Field(description="Current value hash; null requires absent record")
    value_base64: str | None = Field(description="Replacement bytes in base64; null deletes the record")


class WriteInput(DescribeInput):
    changes: list[Change] = Field(min_length=1)
    reason: str = Field(min_length=1)


class StoreConfig(FunctionBaseConfig):
    root: Path


class DescribeConfig(StoreConfig, name="describe_map"):
    pass


class CloneConfig(StoreConfig, name="clone_map"):
    pass


class ReadConfig(StoreConfig, name="read_records"):
    pass


class WriteConfig(StoreConfig, name="write_records"):
    pass


def function_info(config, method_name, schema):
    method = getattr(MapStore(config.root), method_name)

    def execute(value):
        # py-lmdb disallows overlapping environment handles for the same path.
        with _store_lock:
            return method(**value.model_dump())

    async def invoke(value):
        result = await asyncio.to_thread(execute, value)
        return json.dumps(result)

    # NAT infers execution types from the callable as well as the exposed schema.
    invoke.__annotations__ = {"value": schema, "return": str}
    info = FunctionInfo.from_fn(invoke, input_schema=schema, description=method.__doc__)
    info.metadata = {"mcp_exposed": True}
    return info


@register_function(config_type=DescribeConfig, framework_wrappers=[])
async def describe(config: DescribeConfig, builder: Builder):
    yield function_info(config, "describe_map", DescribeInput)


@register_function(config_type=CloneConfig, framework_wrappers=[])
async def clone(config: CloneConfig, builder: Builder):
    yield function_info(config, "clone_map", CloneInput)


@register_function(config_type=ReadConfig, framework_wrappers=[])
async def read(config: ReadConfig, builder: Builder):
    yield function_info(config, "read_records", ReadInput)


@register_function(config_type=WriteConfig, framework_wrappers=[])
async def write(config: WriteConfig, builder: Builder):
    yield function_info(config, "write_records", WriteInput)
