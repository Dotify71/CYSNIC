# CYSNIC Safety & Limitations (SAFETY.md)

This document is a brutally honest assessment of the system's operational limitations. It intentionally omits marketing terminology in favor of engineering realities. CYSNIC is a **Surveillance-Inspired UI Tracker**, not a certified defense product.

## 1. No Hardware or Environmental Certification
This software has **not** been tested against MIL-STD-810H (environmental/durability) or any hardware reliability standards. It is a software-only implementation currently designed for consumer-grade webcams.

## 2. No Re-Detection Capability
CYSNIC relies purely on correlation filters (MIL/KCF) and an Eigen Kalman filter for localized tracking. If a target completely exits the frame and re-enters, or if the tracker drifts significantly onto a background object without triggering the covariance physics gate, **there is no independent detector to re-anchor the target**. It relies solely on local rotation recovery and coasting. True robust systems use a detector (like YOLO) paired with a tracker (like DeepSORT).

## 3. "X-RAY" is Coasting, Not True Occlusion Sensing
The HUD refers to occlusion recovery as "X-RAY". This is purely aesthetic terminology. The system **does not** possess thermal, LIDAR, or through-wall sensing capabilities. When a target is occluded, the Kalman filter simply coasts along the last known velocity vector until the physics gate accepts a new visual match.

## 4. Single Sensor Dependency
This system operates on a single visual spectrum camera. It does not perform sensor fusion. It is highly susceptible to adversarial lighting, total darkness, and heavy visual obstruction.

## 5. Security & Verification
This project is an open-source demonstration of engineering rigor (RAII, Dependency Injection, deterministic testing, mathematical assertions). It has **not** undergone DO-178C formal verification, IV&V (Independent Verification and Validation), or any RMF/ATO security accreditation. Do not deploy this in life-safety or classified environments.
