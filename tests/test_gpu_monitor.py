"""Tests for the GPU monitor's nvidia-smi XML parser."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from cserve.node_agent.gpu_monitor import GpuMonitor, _parse_mib, _parse_pct, _parse_temp, _xml_text

SAMPLE_NVIDIA_SMI_XML = """<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">
<nvidia_smi_log>
<attached_gpus>2</attached_gpus>
<gpu id="GPU-abc123">
  <minor_number>0</minor_number>
  <product_name>NVIDIA A40</product_name>
  <uuid>GPU-abc123</uuid>
  <fb_memory_usage>
    <total>46068 MiB</total>
    <used>1024 MiB</used>
  </fb_memory_usage>
  <utilization>
    <gpu_util>35 %</gpu_util>
  </utilization>
  <temperature>
    <gpu_temp>55 C</gpu_temp>
  </temperature>
</gpu>
<gpu id="GPU-def456">
  <minor_number>1</minor_number>
  <product_name>NVIDIA A40</product_name>
  <uuid>GPU-def456</uuid>
  <fb_memory_usage>
    <total>46068 MiB</total>
    <used>0 MiB</used>
  </fb_memory_usage>
  <utilization>
    <gpu_util>0 %</gpu_util>
  </utilization>
  <temperature>
    <gpu_temp>28 C</gpu_temp>
  </temperature>
</gpu>
</nvidia_smi_log>"""


class TestNvidiaSmiParser:
    def test_parses_two_gpus(self):
        gpus = GpuMonitor._parse_nvidia_smi_xml(SAMPLE_NVIDIA_SMI_XML)
        assert len(gpus) == 2

    def test_gpu0_fields(self):
        gpus = GpuMonitor._parse_nvidia_smi_xml(SAMPLE_NVIDIA_SMI_XML)
        g0 = gpus[0]
        assert g0.index == 0
        assert g0.uuid == "GPU-abc123"
        assert g0.name == "NVIDIA A40"
        assert g0.memory_used_mb == 1024
        assert g0.memory_total_mb == 46068
        assert g0.utilization_pct == 35.0
        assert g0.temperature_c == 55.0

    def test_gpu1_fields(self):
        gpus = GpuMonitor._parse_nvidia_smi_xml(SAMPLE_NVIDIA_SMI_XML)
        g1 = gpus[1]
        assert g1.index == 1
        assert g1.memory_used_mb == 0
        assert g1.utilization_pct == 0.0
        assert g1.temperature_c == 28.0

    def test_invalid_xml_returns_empty(self):
        gpus = GpuMonitor._parse_nvidia_smi_xml("not xml")
        assert gpus == []


class TestXmlHelpers:
    def test_xml_text_default(self):
        root = ET.fromstring("<a><b>hello</b></a>")
        assert _xml_text(root, "b") == "hello"
        assert _xml_text(root, "c", "default") == "default"

    def test_parse_mib(self):
        root = ET.fromstring("<mem><used>1024 MiB</used></mem>")
        assert _parse_mib(root, "used") == 1024

    def test_parse_pct(self):
        root = ET.fromstring("<util><gpu_util>42 %</gpu_util></util>")
        assert _parse_pct(root, "gpu_util") == 42.0

    def test_parse_temp(self):
        root = ET.fromstring("<temp><gpu_temp>67 C</gpu_temp></temp>")
        assert _parse_temp(root, "gpu_temp") == 67.0
