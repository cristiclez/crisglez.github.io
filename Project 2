# 🎓 EventFlow Tracker: Plantilla de Gestión Logística y Financiera para Eventos

A pragmatic, data-driven Excel workbook designed to orchestrate event logistics, cash flows, and resource allocation without the operational overhead of complex architectures.

---

## 🧠 Overview

This repository delivers a reliable and straightforward ledger system to manage the financial lifecycle of large group events. 

Originally built to coordinate the graduation events of the **Mathematics Promotion (2021-2026) at the University of Oviedo**, the spreadsheet successfully handled near **€10,000** in fractional cash inflows, multi-tier ticket pricing (students vs. faculty), and transport capacity constraints.

> 💬 *"Utilizamos mucho Power BI en nuestro día a día, pero a veces nos olvidamos del Excel y de lo útil que es para cosas sencillas como esta: un documento que no hay que presentarle a nadie, es para nosotros, pero nos ayuda a seguir detalladamente cómo va avanzando la planificación."*

---

## 📊 Spreadsheet Structure

The workbook is designed across three simple, interconnected operational areas:

* **Registro General (Alumnos y Acompañantes):** Main ledger tracking attendance packages (Dinner + Party, Party Only), bus slot allocations, and personal deposit statuses.
* **Recuento de Profesores:** Institutional attendance matrix segmented by academic departments (*Física, Estadística, Matemáticas*) featuring specific subsidized pricing tiers.
* **Resumen Financiero (KPI Dashboard):** A single-view control card evaluating real-time health metrics of the event.

---

## 🛠️ Design Choices for Data Integrity

* **Strict Data Validation:** Prevents manual entry errors (e.g., mixing "Si", "SI", and "X"). Using explicit data validation lists and checkboxes guarantees that downstream formulas (`SUMIF`, `COUNTIF`) run with **100% data integrity**.
* **Decoupled Variables:** Unit costs (Dinner: €94.50, Party: €50.00, Bus: €12.00) are fully isolated using absolute cell references (`$A$1`). If a vendor changes a price at the last minute, updating one single cell recalculates the entire ledger instantly.
* **Conditional Alert Heat-maps:** Outstanding accounts receivable remain visually highlighted in warning tones until the payment confirmation checkbox is ticked, drastically reducing cognitive load when checking raw bank statements.

---

## 📈 Output Metrics & KPIs Monitored

The system evaluates the following metrics dynamically:

| KPI Metric | Logic Description | Case Study Values |
| :--- | :--- | :--- |
| **Total Expected Revenue** | Product of multi-tier packages across total registrations. | **€9,864.00** |
| **Realized Capital** | Cumulative sum of confirmed bank transfers. | **€5,526.00** |
| **Accounts Receivable** | Critical bottleneck metric: `Expected - Realized`. | **€4,338.00** |
| **Logistics Capacity** | Dynamic volume boundary control for transportation hiring. | **38 / 72 Bus Seats** |
