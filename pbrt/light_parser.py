import re

from pbrt.parse_utils import parse_value


def parse_light_property_line(line):
    m_decl = re.search(r'"([^"]+)"', line)
    if not m_decl:
        return None, None, None
    declaration = m_decl.group(1)
    parts = declaration.split()
    if len(parts) < 2:
        return None, None, None
    ptype, key = parts[0], parts[1]
    m_val = re.search(r'\[([^\]]+)\]', line)
    if not m_val:
        return ptype, key, None
    raw_value = f"[{m_val.group(1)}]"
    value = parse_value(ptype, raw_value)
    return ptype, key, value

def parse_pbrt_lights_and_counts(filename):
    lights = []
    light_counts = {
        "total_lights": 0,
        "point": 0,
        "spot": 0,
        "distant": 0,
        "infinite": 0,
        "goniometric": 0,
        "projection": 0,
        "area": 0,
    }
    is_area_light = False
    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        if not stripped_line:
            i += 1
            continue
        if stripped_line.startswith("AttributeBegin"):
            is_area_light = True
            i += 1
            continue
        if stripped_line.startswith("AttributeEnd"):
            is_area_light = False
            i += 1
            continue
        if stripped_line.startswith("LightSource") or stripped_line.startswith("AreaLightSource"):
            tokens = stripped_line.split()
            if len(tokens) < 2:
                i += 1
                continue
            light_type = tokens[1].strip('"')
            if is_area_light:
                light_counts["area"] += 1
            elif light_type in light_counts:
                light_counts[light_type] += 1
            light_counts["total_lights"] += 1
            light_data = {
                "type": light_type,
                "is_area": is_area_light,
                "from": [0.0, 0.0, 0.0],
                "to": [0.0, 0.0, 1.0],
                "I": [1.0, 1.0, 1.0],
                "L": [1.0, 1.0, 1.0],
                "scale": 1.0,
                "power": 0.0,
                "illuminance": 0.0,
                "filename": "",
                "coneangle": 30.0,
                "conedeltaangle": 5.0,
                "twosided": False,
                "extra": {}
            }
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith('"'):
                ptype, key, value = parse_light_property_line(lines[i])
                if key is None:
                    i += 1
                    continue
                if key in light_data:
                    light_data[key] = value
                else:
                    light_data["extra"][key] = value
                i += 1
            lights.append(light_data)
        else:
            i += 1
    return lights, light_counts