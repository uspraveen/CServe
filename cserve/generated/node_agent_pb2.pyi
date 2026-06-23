from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LaunchReplicaRequest(_message.Message):
    __slots__ = ("replica_id", "model_name", "served_model_name", "hf_model", "variant", "tp_size", "gpu_ids", "port", "engine_args", "env_vars")
    class EngineArgsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    SERVED_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    HF_MODEL_FIELD_NUMBER: _ClassVar[int]
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    TP_SIZE_FIELD_NUMBER: _ClassVar[int]
    GPU_IDS_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    ENGINE_ARGS_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    replica_id: str
    model_name: str
    served_model_name: str
    hf_model: str
    variant: str
    tp_size: int
    gpu_ids: _containers.RepeatedScalarFieldContainer[int]
    port: int
    engine_args: _containers.ScalarMap[str, str]
    env_vars: _containers.ScalarMap[str, str]
    def __init__(self, replica_id: _Optional[str] = ..., model_name: _Optional[str] = ..., served_model_name: _Optional[str] = ..., hf_model: _Optional[str] = ..., variant: _Optional[str] = ..., tp_size: _Optional[int] = ..., gpu_ids: _Optional[_Iterable[int]] = ..., port: _Optional[int] = ..., engine_args: _Optional[_Mapping[str, str]] = ..., env_vars: _Optional[_Mapping[str, str]] = ...) -> None: ...

class LaunchReplicaResponse(_message.Message):
    __slots__ = ("ok", "error", "replica_id", "http_endpoint", "pid")
    OK_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    HTTP_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    error: str
    replica_id: str
    http_endpoint: str
    pid: int
    def __init__(self, ok: bool = ..., error: _Optional[str] = ..., replica_id: _Optional[str] = ..., http_endpoint: _Optional[str] = ..., pid: _Optional[int] = ...) -> None: ...

class StopReplicaRequest(_message.Message):
    __slots__ = ("replica_id", "force")
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    replica_id: str
    force: bool
    def __init__(self, replica_id: _Optional[str] = ..., force: bool = ...) -> None: ...

class StopReplicaResponse(_message.Message):
    __slots__ = ("ok", "error")
    OK_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    error: str
    def __init__(self, ok: bool = ..., error: _Optional[str] = ...) -> None: ...

class DrainReplicaRequest(_message.Message):
    __slots__ = ("replica_id", "timeout_s")
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    replica_id: str
    timeout_s: int
    def __init__(self, replica_id: _Optional[str] = ..., timeout_s: _Optional[int] = ...) -> None: ...

class DrainReplicaResponse(_message.Message):
    __slots__ = ("ok", "error", "drained_requests")
    OK_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    DRAINED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    error: str
    drained_requests: int
    def __init__(self, ok: bool = ..., error: _Optional[str] = ..., drained_requests: _Optional[int] = ...) -> None: ...

class PingRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PingResponse(_message.Message):
    __slots__ = ("node_name", "hostname", "uptime_s")
    NODE_NAME_FIELD_NUMBER: _ClassVar[int]
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    UPTIME_S_FIELD_NUMBER: _ClassVar[int]
    node_name: str
    hostname: str
    uptime_s: float
    def __init__(self, node_name: _Optional[str] = ..., hostname: _Optional[str] = ..., uptime_s: _Optional[float] = ...) -> None: ...

class NodeStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class NodeStatusResponse(_message.Message):
    __slots__ = ("node_name", "gpus", "replicas", "uptime_s", "load_avg_1m", "ram_used_mb", "ram_total_mb")
    NODE_NAME_FIELD_NUMBER: _ClassVar[int]
    GPUS_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    UPTIME_S_FIELD_NUMBER: _ClassVar[int]
    LOAD_AVG_1M_FIELD_NUMBER: _ClassVar[int]
    RAM_USED_MB_FIELD_NUMBER: _ClassVar[int]
    RAM_TOTAL_MB_FIELD_NUMBER: _ClassVar[int]
    node_name: str
    gpus: _containers.RepeatedCompositeFieldContainer[GpuInfo]
    replicas: _containers.RepeatedCompositeFieldContainer[ReplicaInfo]
    uptime_s: float
    load_avg_1m: float
    ram_used_mb: int
    ram_total_mb: int
    def __init__(self, node_name: _Optional[str] = ..., gpus: _Optional[_Iterable[_Union[GpuInfo, _Mapping]]] = ..., replicas: _Optional[_Iterable[_Union[ReplicaInfo, _Mapping]]] = ..., uptime_s: _Optional[float] = ..., load_avg_1m: _Optional[float] = ..., ram_used_mb: _Optional[int] = ..., ram_total_mb: _Optional[int] = ...) -> None: ...

class GpuInfo(_message.Message):
    __slots__ = ("index", "uuid", "name", "memory_used_mb", "memory_total_mb", "utilization_pct", "temperature_c", "state", "allocated_replica_id")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USED_MB_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_MB_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_PCT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_C_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ALLOCATED_REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    index: int
    uuid: str
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_pct: float
    temperature_c: float
    state: str
    allocated_replica_id: str
    def __init__(self, index: _Optional[int] = ..., uuid: _Optional[str] = ..., name: _Optional[str] = ..., memory_used_mb: _Optional[int] = ..., memory_total_mb: _Optional[int] = ..., utilization_pct: _Optional[float] = ..., temperature_c: _Optional[float] = ..., state: _Optional[str] = ..., allocated_replica_id: _Optional[str] = ...) -> None: ...

class ReplicaInfo(_message.Message):
    __slots__ = ("replica_id", "model_name", "variant", "gpu_ids", "port", "pid", "status", "health_ok", "uptime_s")
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    GPU_IDS_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    HEALTH_OK_FIELD_NUMBER: _ClassVar[int]
    UPTIME_S_FIELD_NUMBER: _ClassVar[int]
    replica_id: str
    model_name: str
    variant: str
    gpu_ids: _containers.RepeatedScalarFieldContainer[int]
    port: int
    pid: int
    status: str
    health_ok: bool
    uptime_s: float
    def __init__(self, replica_id: _Optional[str] = ..., model_name: _Optional[str] = ..., variant: _Optional[str] = ..., gpu_ids: _Optional[_Iterable[int]] = ..., port: _Optional[int] = ..., pid: _Optional[int] = ..., status: _Optional[str] = ..., health_ok: bool = ..., uptime_s: _Optional[float] = ...) -> None: ...

class ReplicaStatusRequest(_message.Message):
    __slots__ = ("replica_id",)
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    replica_id: str
    def __init__(self, replica_id: _Optional[str] = ...) -> None: ...

class ReplicaStatusResponse(_message.Message):
    __slots__ = ("replica", "vllm_metrics")
    class VllmMetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    REPLICA_FIELD_NUMBER: _ClassVar[int]
    VLLM_METRICS_FIELD_NUMBER: _ClassVar[int]
    replica: ReplicaInfo
    vllm_metrics: _containers.ScalarMap[str, float]
    def __init__(self, replica: _Optional[_Union[ReplicaInfo, _Mapping]] = ..., vllm_metrics: _Optional[_Mapping[str, float]] = ...) -> None: ...

class KillGpuProcessesRequest(_message.Message):
    __slots__ = ("gpu_ids", "vllm_only")
    GPU_IDS_FIELD_NUMBER: _ClassVar[int]
    VLLM_ONLY_FIELD_NUMBER: _ClassVar[int]
    gpu_ids: _containers.RepeatedScalarFieldContainer[int]
    vllm_only: bool
    def __init__(self, gpu_ids: _Optional[_Iterable[int]] = ..., vllm_only: bool = ...) -> None: ...

class KillGpuProcessesResponse(_message.Message):
    __slots__ = ("ok", "error", "killed_pids")
    OK_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    KILLED_PIDS_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    error: str
    killed_pids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, ok: bool = ..., error: _Optional[str] = ..., killed_pids: _Optional[_Iterable[int]] = ...) -> None: ...
