# Energy Measurement Methodology & Limitations

## 1. Energy Telemetry Scope

Energy measurement in edge and federated learning systems assesses the energy consumption (in Joules or Watt-hours) incurred during local model training on edge nodes, encryption operations, and cloud aggregation.

---

## 2. Experimental Environment & Hardware Inspection

Telemetry inspection was conducted on the following environment:
- **Platform:** Windows 11 (AMD64 / x86_64)
- **CPU Architecture:** Intel Core Processor (4 logical cores)
- **Power Telemetry Support:** Hardware RAPL (Running Average Power Limit) counters and `sysfs` power interfaces are unavailable under standard Windows user permissions without low-level ring-0 hardware drivers or physical power meters.

---

## 3. Mandatory Energy Telemetry Status & Statement

> **"Validated joule-level energy telemetry was not available in the current Windows execution environment, so energy consumption is reported as unavailable rather than estimated as measured energy."**

---

## 4. Methodological Guidelines for Energy Reporting

To ensure strict empirical accuracy and compliance with project hard rules:
1. **No Fabricated Telemetry:** Synthetic energy numbers or unvalidated physical measurements are strictly prohibited.
2. **CPU-Time x TDP Rule:** If a theoretical CPU-time multiplied by Thermal Design Power (TDP) calculation is referenced in comparative analyses, it must be explicitly labeled as an **"estimate, not measured"** calculation rather than true physical energy consumption.
3. **Current Reporting Value:** Energy consumption across all local training runs, encryption routines, and FL rounds is recorded as `N/A`.
