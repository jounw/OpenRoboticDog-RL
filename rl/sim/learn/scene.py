"""Build the training MJCF: strip meshes from the robot model (keeping inertia as explicit values), leaving only the floor, body boxes and foot spheres.

MJX does not need to carry 113 visual meshes. The full model is compiled once and each body's mass, center of mass and inertia are
read and written into <inertial>, so the dynamics match the original model.

Additions:
  - sensors: upvector(framezaxis imu), local_linvel(velocimeter imu), global_linvel/global_angvel(imu),
    foot site position/velocity (framepos/framelinvel), foot-floor contact (contact found)
  - foot geoms priority=1, contype=1 conaffinity=0 -> foot friction determines contact friction; feet do not collide with each other or the body
  - floor contype=1 conaffinity=1
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROBOT_XML = Path(__file__).resolve().parents[1] / "assets" / "openroboticdog" / "openroboticdog.xml"
FEET = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
JOINTS = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]


def load_full_model(xml_path=ROBOT_XML):
    xml_path = Path(xml_path)
    assets = {p.name: p.read_bytes() for p in (xml_path.parent / "meshes").glob("*.stl")}
    return mujoco.MjModel.from_xml_string(xml_path.read_text(encoding="utf-8"), assets)


def _fmt(v):
    return " ".join(f"{x:.8g}" for x in np.asarray(v).ravel())


def build_training_xml(xml_path=ROBOT_XML):
    """Training MJCF string. No meshes, explicit inertia, sensors added."""
    full = load_full_model(xml_path)
    root = ET.fromstring(Path(xml_path).read_text(encoding="utf-8"))

    # remove mesh assets
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        asset.remove(mesh)

    # per body: remove visual geoms, write compiled inertia as <inertial>
    for body in root.iter("body"):
        name = body.get("name")
        for geom in list(body.findall("geom")):
            if geom.get("class") == "visual":
                body.remove(geom)
        bid = full.body(name).id
        inertial = ET.Element("inertial", {
            "pos": _fmt(full.body_ipos[bid]), "quat": _fmt(full.body_iquat[bid]),
            "mass": f"{full.body_mass[bid]:.8g}", "diaginertia": _fmt(full.body_inertia[bid]),
        })
        body.insert(0, inertial)

    # collision settings
    for geom in root.iter("geom"):
        n = geom.get("name")
        if n == "floor":
            geom.set("contype", "1"); geom.set("conaffinity", "1"); geom.set("condim", "3")
            geom.set("friction", "0.8 0.02 0.01")
        elif n in FEET or n == "trunk_box":
            geom.set("contype", "1"); geom.set("conaffinity", "0")
            if n in FEET:
                geom.set("priority", "1")

    # sensors
    sensor = root.find("sensor")
    ET.SubElement(sensor, "framezaxis", {"name": "upvector", "objtype": "site", "objname": "imu"})
    ET.SubElement(sensor, "velocimeter", {"name": "local_linvel", "site": "imu"})
    ET.SubElement(sensor, "framelinvel", {"name": "global_linvel", "objtype": "site", "objname": "imu"})
    ET.SubElement(sensor, "frameangvel", {"name": "global_angvel", "objtype": "site", "objname": "imu"})
    for f in FEET:
        ET.SubElement(sensor, "framepos", {"name": f"{f}_pos", "objtype": "site", "objname": f})
        ET.SubElement(sensor, "framelinvel", {"name": f"{f}_global_linvel", "objtype": "site", "objname": f})

    # visual quality settings are unnecessary for MJX
    vis = root.find("visual")
    if vis is not None:
        root.remove(vis)
    return ET.tostring(root, encoding="unicode")


def load_training_model(xml_path=ROBOT_XML):
    return mujoco.MjModel.from_xml_string(build_training_xml(xml_path))


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROBOT_XML.with_name("openroboticdog_mjx.xml")
    out.write_text(build_training_xml(), encoding="utf-8")
    m = load_training_model()
    print(f"{out}: body {m.nbody}, geom {m.ngeom}, sensor {m.nsensor}, mass {m.body_subtreemass[1]:.3f} kg")
