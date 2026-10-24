# Hardware, wiring and power

## Hardware

| Part | Spec |
|---|---|
| Servos | 12 × Feetech STS3215, 12 V version, stall torque about 2.9 N·m, IDs 1–12 |
| Joint order | FR, FL, RR, RL legs × hip, thigh, calf (FR = IDs 1–3, FL = 4–6, RR = 7–9, RL = 10–12) |
| Servo driver | Waveshare Bus Servo Driver HAT (A), mounted on the Pi 40-pin header |
| Computer | Raspberry Pi 5 (runs the policy), or a laptop connected directly to the HAT USB |
| IMU | BNO055 |
| Body | 3D-printed (original STL files, PETG) |
| Mass | 1.497 kg total, battery included |

## Wiring and power

| Item | Spec |
|---|---|
| Battery | 3S LiPo 2200 mAh, 11.1 V nominal, 11.8–12.4 V measured |
| Bench supply | 12 V 30 A DC supply for development. Use the battery or the supply, never both at once |
| Power input | HAT green screw terminal (V = +, G = −), 16 AWG wire. The HAT DC jack is in parallel with it |
| Fuse | Inline 10 A (15 A for walking tests) |
| Servo buses | HAT TTL port A: front legs (IDs 1–6). Port B: rear legs (IDs 7–12). 12 V, 1 Mbps, daisy-chained |
| Servo link | HAT servo USB-C (CH343 USB-UART) to Pi USB-A, HAT switch set to USB. Appears as `/dev/ttyACM0` on the Pi and `/dev/cu.usbmodem*` on macOS. The ESP32 / GPIO UART path on the HAT is not used |
| Pi power | 5 V from the HAT through the 40-pin header. No undervoltage flags (`vcgencmd get_throttled` = 0x0) |
| IMU wiring | BNO055 VIN to Pi pin 1 (3.3 V), SDA to pin 3, SCL to pin 5, GND to pin 6. I2C bus 1, address 0x28 |
| Cooling | Pi fan forced to maximum speed |
| Measured current | Standing, all 12 servos: up to 0.58 A. One servo in PWM mode at duty 800, in the air: up to 2.1 A |
| Software cutoffs | Below 10.5 V, or above 2.5 A or 65 °C on any servo (2 consecutive readings) |
| Network | Phone web joystick over Wi-Fi on port 8080. The Pi can also run its own 5 GHz hotspot for low latency |
| Not measured yet | Total current while walking |

Servo measurements (PWM model, PD step response, registers) are in [`../sim/sysid/SERVO_DATASHEET.md`](../sim/sysid/SERVO_DATASHEET.md).
