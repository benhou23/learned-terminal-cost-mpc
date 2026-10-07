"""Writes the SUMO network, routes and config for a straight 3-lane highway."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_oneway_highway_3lane(
    out_dir: str | Path,
    *,
    length_m: float = 5000.0,
    speed_mps: float = 33.33,
    sim_step: float = 0.1,
    duration_s: int = 3600,
    car_vph: int = 0,
    truck_vph: int = 0,
    ego_depart_s: float = 0.0,
) -> dict:

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    nodes_xml = f"""<nodes>
  <node id="n0" x="0.0" y="0.0" type="priority"/>
  <node id="n1" x="{length_m:.1f}" y="0.0" type="priority"/>
</nodes>
"""
    nodes_path = out / "nodes.nod.xml"
    _write_text(nodes_path, nodes_xml)

    edges_xml = f"""<edges>
  <edge id="e0" from="n0" to="n1" numLanes="3" speed="{speed_mps:.2f}"/>
</edges>
"""
    edges_path = out / "edges.edg.xml"
    _write_text(edges_path, edges_xml)

    flow_car = ""
    if int(car_vph) > 0:
        flow_car = f"""
  <flow id="car_flow"
        type="car"
        route="r0"
        begin="0" end="{duration_s}"
        vehsPerHour="{int(car_vph)}"
        departLane="free"
        departSpeed="max"/>"""

    flow_truck = ""
    if int(truck_vph) > 0:
        flow_truck = f"""
  <flow id="truck_flow"
        type="truck"
        route="r0"
        begin="0" end="{duration_s}"
        vehsPerHour="{int(truck_vph)}"
        departLane="free"
        departSpeed="max"/>"""

    routes_xml = f"""<routes>
  <vType id="car"
         vClass="passenger"
         accel="2.6" decel="4.5"
         length="4.8" minGap="2.5"
         maxSpeed="{speed_mps:.2f}"
         sigma="0.5" tau="1.0"/>

  <vType id="truck"
         vClass="truck"
         accel="1.2" decel="4.0"
         length="16.5" minGap="3.0"
         maxSpeed="{speed_mps:.2f}"
         sigma="0.5" tau="1.0"/>

  <vType id="egoTruck"
         vClass="truck"
         accel="1.0" decel="4.0"
         sigma="0.0" length="12.0" maxSpeed="{speed_mps:.2f}"/>

  <route id="r0" edges="e0"/>{flow_car}{flow_truck}
</routes>
"""
    routes_path = out / "routes.rou.xml"
    _write_text(routes_path, routes_xml)

    sumocfg_xml = f"""<configuration>
  <input>
    <net-file value="net.net.xml"/>
    <route-files value="routes.rou.xml"/>
  </input>

  <time>
    <begin value="0"/>
    <end value="{duration_s}"/>
    <step-length value="{sim_step}"/>
  </time>

  <report>
    <verbose value="true"/>
    <no-warnings value="true"/>
  </report>
</configuration>
"""
    sumocfg_path = out / "run.sumocfg"
    _write_text(sumocfg_path, sumocfg_xml)

    net_path = out / "net.net.xml"
    if not net_path.exists():
        try:
            subprocess.run(
                ["netconvert", "-n", str(nodes_path), "-e", str(edges_path),
                 "-o", str(net_path)],
                timeout=30,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("netconvert timed out after 30s")

    return {
        "dir": str(out),
        "nodes": str(nodes_path),
        "edges": str(edges_path),
        "net": str(net_path),
        "routes": str(routes_path),
        "sumocfg": str(sumocfg_path),
    }
