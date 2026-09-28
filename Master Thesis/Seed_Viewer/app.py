from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeHalfSpace, BRepPrimAPI_MakePrism
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pln, gp_Pnt, gp_Vec

HERE = Path(__file__).resolve().parent
SEED_PATH = HERE.parent / "Seed_Extractor" / "data" / "feature_space_seeds.json"
SEEDS = json.loads(SEED_PATH.read_text())
FEATURES = {f["name"]: f for f in SEEDS["features"]}
LIMIT = 1.5
GREEN = "#2dbe69"
RED = "#e12d2d"
HOST, PORT = "127.0.0.1", 8051


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def plane(n, p):
    n = unit(n)
    return n, float(n @ p)


def clip(poly, n, d):
    result = []
    if not len(poly):
        return result
    for a, b in zip(poly, list(poly[1:]) + [poly[0]]):
        fa, fb = n @ a - d, n @ b - d
        if fa >= -1e-9:
            result.append(a)
        if (fa > 1e-9 and fb < -1e-9) or (fa < -1e-9 and fb > 1e-9):
            result.append(a + fa / (fa - fb) * (b - a))
    return result


def box_planes(lo, hi):
    return [(np.eye(3)[i], lo[i]) for i in range(3)] + [
        (-np.eye(3)[i], -hi[i]) for i in range(3)]


def plane_polygon(n, d, constraints):
    helper = [1, 0, 0] if abs(n[0]) < .8 else [0, 1, 0]
    u = unit(np.cross(n, helper))
    v = np.cross(n, u)
    p = n * d
    poly = [p + a*u + b*v for a, b in [(-20,-20),(20,-20),(20,20),(-20,20)]]
    for cn, cd in constraints:
        poly = clip(poly, cn, cd)
    return np.asarray(poly)


def mesh(vertices, triangles, name, red=False):
    v = np.asarray(vertices)
    return go.Mesh3d(
        x=v[:,0], y=v[:,1], z=v[:,2],
        i=[t[0] for t in triangles], j=[t[1] for t in triangles], k=[t[2] for t in triangles],
        color=RED if red else GREEN, opacity=.65 if red else .34,
        flatshading=False, name=name, showlegend=False,
        lighting=dict(ambient=.55, diffuse=.8, specular=.15),
        hovertemplate=name+"<extra></extra>")

def boundary_line(points, name, close=False):
    p = np.asarray(points, dtype=float)

    if len(p) < 2:
        return None

    if close and np.linalg.norm(p[0] - p[-1]) > 1e-9:
        p = np.vstack([p, p[0]])

    return go.Scatter3d(
        x=p[:, 0],
        y=p[:, 1],
        z=p[:, 2],
        mode="lines",
        line=dict(color="#FFD700", width=6),
        name=name + " trim boundary",
        showlegend=False,
        hoverinfo="skip",
    )

def wall_boundaries(path, z0, z1, name):
    path = np.asarray(path)

    bottom = np.column_stack([
        path[:, 0],
        path[:, 1],
        np.full(len(path), z0)
    ])

    top = np.column_stack([
        path[:, 0],
        path[:, 1],
        np.full(len(path), z1)
    ])

    result = [
        boundary_line(bottom, name),
        boundary_line(top, name),
    ]

    # 如果 path 不是闭合 curve，
    # 两个端点还各有一条 vertical trim boundary
    if np.linalg.norm(path[0] - path[-1]) > 1e-9:
        result.append(boundary_line([
            [path[0, 0], path[0, 1], z0],
            [path[0, 0], path[0, 1], z1],
        ], name))

        result.append(boundary_line([
            [path[-1, 0], path[-1, 1], z0],
            [path[-1, 0], path[-1, 1], z1],
        ], name))

    return result


def polygon_mesh(poly, name, red=False):
    clean=[]
    for p in poly:
        if not clean or np.linalg.norm(np.asarray(p)-clean[-1])>1e-9:
            clean.append(np.asarray(p))
    if len(clean)>1 and np.linalg.norm(clean[0]-clean[-1])<1e-9:
        clean.pop()
    return mesh(clean, [(0,i,i+1) for i in range(1,len(clean)-1)], name, red)


def polygon_area(poly):
    if len(poly)<3:
        return 0.
    return .5*np.linalg.norm(sum((np.cross(poly[i]-poly[0],poly[i+1]-poly[0]) for i in range(1,len(poly)-1)),np.zeros(3)))


def planar_spec(name):
    lo, hi = np.array([-.65,-.65,-.65]), np.array([.65,.65,.65])
    opens = [(False,False)]*3
    if "passage" in name or name in {"triangular_pocket","rectangular_pocket","6sides_pocket"}:
        count = 3 if name.startswith("triangular") else 6 if name.startswith("6sides") else 4
        if count == 4:
            points = np.array([[-.65,-.5],[.65,-.5],[.65,.5],[-.65,.5]])
        else:
            points = np.array([[.8*math.cos(2*math.pi*i/count),.8*math.sin(2*math.pi*i/count)] for i in range(count)])
        planes = []
        for a,b in zip(points,np.roll(points,-1,axis=0)):
            n = unit([-(b-a)[1], (b-a)[0], 0])
            planes.append(plane(n, [*a,0]))
        lo[:2],hi[:2] = points.min(axis=0),points.max(axis=0)
        pocket = "pocket" in name
        if pocket:
            planes.append(plane([0,0,1],[0,0,lo[2]]))
        opens = [(False,False),(False,False),(not pocket,True)]
    elif name == "triangular_through_slot":
        lo,hi = np.array([-.65,-.5,-.65]),np.array([.65,.65,.65])
        planes = [plane([1.15,.65,0],[0,-.5,0]),plane([-1.15,.65,0],[0,-.5,0])]
        opens = [(True,True),(False,True),(True,True)]
    elif name == "rectangular_through_slot":
        planes = [plane([1,0,0],lo),plane([0,1,0],lo),plane([-1,0,0],hi)]
        opens = [(False,False),(False,True),(True,True)]
    elif name == "rectangular_through_step":
        planes = [plane([0,1,0],lo),plane([1,0,0],lo)]
        opens = [(False,True),(False,True),(True,True)]
    elif name == "2sides_through_step":
        lo,hi = np.array([-.65,-.5,-.65]),np.array([.65,.65,.65])
        planes = [plane([1.15,.65,0],[0,-.5,0]),plane([-1.15,.65,0],[0,-.5,0]),plane([0,0,1],lo)]
        opens = [(True,True),(False,True),(False,True)]
    elif name == "rectangular_blind_slot":
        planes = [plane([1,0,0],lo),plane([0,1,0],lo),plane([-1,0,0],hi),plane([0,0,1],lo)]
        opens = [(False,False),(False,True),(False,True)]
    elif name == "rectangular_blind_step":
        planes = [plane([1,0,0],lo),plane([0,1,0],lo),plane([0,0,1],lo)]
        opens = [(False,True)]*3
    else:
        return None
    return planes,lo,hi,opens


def planar_layers(feature):
    planes,lo,hi,opens = planar_spec(feature["name"])
    layers = {"surfaces":[], "trimmed":[], "boundaries":[], "supports":{}}
    cube = box_planes([-LIMIT]*3,[LIMIT]*3)
    bounds = box_planes(lo,hi)
    for i,(n,d) in enumerate(planes):
        layers["supports"][feature["nodes"][i]["id"]] = dict(type="plane", normal=n, offset=d)
        label = f"S{feature['nodes'][i]['id']} · {feature['nodes'][i]['role']}"
        layers["surfaces"].append(polygon_mesh(plane_polygon(n,d,cube),label))
        poly = plane_polygon(n,d,bounds+planes)
        if len(poly) < 3:
            raise ValueError(f"Inactive boundary: {feature['name']} {label}")
        layers["trimmed"].append(polygon_mesh(poly,label))
        layers["boundaries"].append(boundary_line(poly, label, close=True))
    return layers


def arc(cx,cy,r,a,b):
    theta = np.linspace(a,b,65)
    return np.column_stack([cx+r*np.cos(theta),cy+r*np.sin(theta)])


def wall(path,z0,z1,name,red=False):
    v = [[x,y,z] for z in (z0,z1) for x,y in path]
    k = len(path)
    triangles = [(i,i+1,k+i+1) for i in range(k-1)] + [(i,k+i+1,k+i) for i in range(k-1)]
    return mesh(v,triangles,name,red)


def annular_cap(ro,ri,z,name,red=False):
    outer,inner = arc(0,0,ro,0,2*math.pi),arc(0,0,ri,0,2*math.pi)
    k = len(outer)
    v = [[x,y,z] for x,y in np.concatenate([outer,inner])]
    ts = [(i,i+1,k+i+1) for i in range(k-1)]+[(i,k+i+1,k+i) for i in range(k-1)]
    return mesh(v,ts,name,red)


def curved_layers(feature):
    name=feature["name"]
    layers = {"surfaces":[], "trimmed":[],"boundaries": [], "supports":{}}
    r,c=.4,.5
    z0,z1=-.65,.65
    blind=name not in {"through_hole","circular_through_slot"}
    # Entries are seed roles, each containing one finite boundary path.
    # Curves retain a single cylinder node regardless of tessellation density.
    if name == "circular_end_pocket":
        paths=[np.array([[-c,-r],[c,-r]]),arc(c,0,r,-math.pi/2,math.pi/2),
               np.array([[c,r],[-c,r]]),arc(-c,0,r,math.pi/2,3*math.pi/2)]
        cylinders={1:(c,0,r),3:(-c,0,r)}
        contour=np.concatenate(paths)
    elif name == "v_circular_end_blind_slot":
        paths=[np.array([[c,r],[-c,r]]),arc(-c,0,r,math.pi/2,3*math.pi/2),
               np.array([[-c,-r],[c,-r]])]
        cylinders={1:(-c,0,r)}
        contour=np.concatenate(paths)
    elif name == "h_circular_end_blind_slot":
        paths=[arc(-c,0,r,math.pi,1.5*math.pi),np.array([[-c,-r],[c,-r]]),
               arc(c,0,r,1.5*math.pi,2*math.pi)]
        cylinders={0:(-c,0,r),2:(c,0,r)}
        contour=np.concatenate(paths)
    else:
        r=.62
        partial=name in {"circular_through_slot","circular_blind_step"}
        path=arc(0,0,r,0,math.pi if partial else 2*math.pi)
        paths=[path]
        cylinders={0:(0,0,r)}
        contour=arc(0,0,r,0,2*math.pi)
        if name == "circular_blind_step":
            contour=path
    ring=name=="Oring"
    if ring:
        paths=[arc(0,0,.62,0,2*math.pi),None,arc(0,0,.30,0,2*math.pi)]
        cylinders={0:(0,0,.62),2:(0,0,.30)}
    nodes=feature["nodes"]
    for i,node in enumerate(nodes):
        label=f"S{node['id']} · {node['role']}"
        if node["surface_type"]=="cylinder":
            cx, cy, rad = cylinders[i]
            layers["supports"][node["id"]] = dict(
                type="cylinder", center=np.array([cx,cy,0.]), radius=rad,
                axis=np.array([0.,0.,1.]), sigma=1 if ring and i==2 else -1,
                path=paths[i], z_bounds=(z0,z1))

            layers["surfaces"].append(
                wall(
                    arc(cx, cy, rad, 0, 2*math.pi),
                    -LIMIT,
                    LIMIT,
                    label
                )
            )

            layers["trimmed"].append(
                wall(paths[i], z0, z1, label)
            )

            layers["boundaries"].extend(
                wall_boundaries(paths[i], z0, z1, label)
            )
        elif node["role"] in {"bottom", "floor", "annular_bottom"}:
            layers["supports"][node["id"]] = dict(type="plane", normal=np.array([0.,0.,1.]), offset=z0)
            layers["surfaces"].append(
                polygon_mesh(
                    plane_polygon(
                        np.array([0,0,1]),
                        z0,
                        box_planes([-LIMIT]*3,[LIMIT]*3)
                    ),
                    label
                )
            )

            if ring:
                layers["trimmed"].append(
                    annular_cap(.62, .30, z0, label)
                )

                # outer trim boundary
                outer = arc(0, 0, .62, 0, 2*math.pi)
                outer3d = np.column_stack([
                    outer[:,0],
                    outer[:,1],
                    np.full(len(outer), z0)
                ])

                # inner trim boundary
                inner = arc(0, 0, .30, 0, 2*math.pi)
                inner3d = np.column_stack([
                    inner[:,0],
                    inner[:,1],
                    np.full(len(inner), z0)
                ])

                layers["boundaries"].append(
                    boundary_line(outer3d, label, close=True)
                )
                layers["boundaries"].append(
                    boundary_line(inner3d, label, close=True)
                )

            else:
                base = paths[0] if name == "circular_blind_step" else contour

                base3d = np.asarray([
                    [x, y, z0]
                    for x, y in base
                ])

                layers["trimmed"].append(
                    polygon_mesh(base3d, label)
                )

                layers["boundaries"].append(
                    boundary_line(base3d, label, close=True)
                )
        else:
            p=paths[i]
            edge=p[-1]-p[0]
            n=unit([-edge[1],edge[0],0])
            d=float(n @ [*p[0],0])
            layers["supports"][node["id"]] = dict(type="plane", normal=n, offset=d)
            layers["surfaces"].append(
                polygon_mesh(
                    plane_polygon(
                        n,
                        d,
                        box_planes([-LIMIT]*3, [LIMIT]*3)
                    ),
                    label
                )
            )

            layers["trimmed"].append(
                wall(p, z0, z1, label)
            )

            layers["boundaries"].extend(
                wall_boundaries(p, z0, z1, label)
            )
    return layers


def build_layers(feature):
    return planar_layers(feature) if planar_spec(feature["name"]) is not None else curved_layers(feature)


def canonical(v):
    v = unit(v)
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


def local_frame(feature, supports):
    """Evaluate JSON frame recipes once from oriented supports, not face extents."""
    recipe = feature["face_constraints"]["feature_space"]["local_crs"]

    def direction(spec):
        kind = spec["type"]
        if kind == "plane_normal":
            return supports[spec["node"]]["normal"]
        if kind == "cylinder_axis":
            return canonical(supports[spec["node"]]["axis"])
        normals = [supports[i]["normal"] for i in spec["nodes"]]
        if kind == "normal_bisector":
            return unit(sum(normals))
        if kind == "intersection_axis":
            axis = canonical(np.cross(*normals))
            if "align_with_normal" in spec and axis @ supports[spec["align_with_normal"]]["normal"] < 0:
                axis = -axis
            return axis
        raise ValueError(f"Unsupported frame recipe: {kind}")

    z = unit(direction(recipe["z"]))
    xspec = recipe["x"]
    x = np.eye(3)[np.argmin(np.abs(z))] if xspec["type"] == "stable_perpendicular" else direction(xspec)
    x = unit(x - (x @ z)*z)
    return np.column_stack([x, unit(np.cross(z,x)), z])


def face_points(layers, frame):
    """Preserve displayed patches; include analytic arc extrema for tight bbox."""
    points = [np.column_stack([t.x,t.y,t.z]) for t in layers["trimmed"]]
    for support in layers["supports"].values():
        if support["type"] != "cylinder":
            continue
        center, radius = support["center"], support["radius"]
        path = support["path"] - center[:2]
        angles = np.unwrap(np.arctan2(path[:,1],path[:,0]))
        low, high = min(angles), max(angles)
        # These demo cylinders are parallel to model z. The CRS may be rotated.
        for axis in frame.T:
            theta = math.atan2(axis[1],axis[0])
            for k in range(-4,5):
                a = theta+k*math.pi
                if low-1e-10 <= a <= high+1e-10:
                    points.append(np.array([[center[0]+radius*math.cos(a),
                                             center[1]+radius*math.sin(a),z]
                                            for z in support["z_bounds"]]))
    return np.concatenate(points)


def prism(poly, vector):
    wire = BRepBuilderAPI_MakePolygon()
    for p in poly:
        wire.Add(gp_Pnt(*map(float,p)))
    wire.Close()
    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    return BRepPrimAPI_MakePrism(face,gp_Vec(*map(float,vector))).Shape()


def oriented_box(frame, low, high):
    # Mathematical CRS origin is the model origin, never the face centroid.
    polygon = [frame @ np.array([x,y,low[2]]) for x,y in
               [(low[0],low[1]),(high[0],low[1]),(high[0],high[1]),(low[0],high[1])]]
    return prism(polygon,frame[:,2]*(high[2]-low[2]))


def boolean(op, a, b):
    operation = op(a,b)
    operation.Build()
    if not operation.IsDone():
        raise RuntimeError("OpenCascade Boolean construction failed")
    return operation.Shape()


def keep_halfspace(shape, normal, offset):
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(*map(float,normal*offset)),
                                         gp_Dir(*map(float,normal)))).Face()
    inside = gp_Pnt(*map(float,normal*(offset+1.)))
    half = BRepPrimAPI_MakeHalfSpace(face,inside).Solid()
    return boolean(BRepAlgoAPI_Common,shape,half)


def cylinder(support, reach):
    axis, center = support["axis"], support["center"]
    return BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(*map(float,center-reach*axis)),gp_Dir(*map(float,axis))),
        float(support["radius"]),2*reach).Shape()


def capsule(a,b,radius,reach):
    """Filled disk (+) center segment, extruded over a safe finite work range."""
    def disk(p):
        return cylinder(dict(axis=np.array([0.,0.,1.]),center=np.array([*p,0.]),radius=radius),reach)
    result = disk(a)
    if np.linalg.norm(b-a) < 1e-10:
        return result
    tangent = unit(b-a)
    normal = np.array([-tangent[1],tangent[0]])*radius
    polygon = [[*p,-reach] for p in (a+normal,b+normal,b-normal,a-normal)]
    result = boolean(BRepAlgoAPI_Fuse,result,prism(polygon,[0,0,2*reach]))
    return boolean(BRepAlgoAPI_Fuse,result,disk(b))


def minkowski_cell(spec, supports, reach):
    section = spec["section"]
    ids = section["generator"]["cylinder_nodes"]
    centers = [supports[i]["center"][:2] for i in ids]
    a, b = centers[0], centers[-1]
    radius = supports[section["disk_radius_from_node"]]["radius"]
    ray = section["opening_ray"]
    if ray is None:
        result = capsule(a,b,radius,reach)
    else:
        if ray["type"] == "plane_normal":
            direction = supports[ray["node"]]["normal"][:2]
        else:
            support = supports[ray["node"]]
            # Fixed reference trim branch, not the current selected patch bbox.
            path = support["path"]
            midpoint = path[len(path)//2] - support["center"][:2]
            direction = support["sigma"]*unit(midpoint)
        v = unit(direction)*(4*reach)
        if len(ids) == 1:
            result = capsule(a,a+v,radius,reach)
        else:
            # (center segment + finite opening segment) + disk.
            # Finite opening endpoint lies well outside every queried box.
            corners = [a,b,b+v,a+v]
            result = prism([[*p,-reach] for p in corners],[0,0,2*reach])
            for p,q in zip(corners,corners[1:]+corners[:1]):
                result = boolean(BRepAlgoAPI_Fuse,result,capsule(p,q,radius,reach))
    bottom = supports[section["bottom_node"]]
    return keep_halfspace(result,bottom["normal"],bottom["offset"])


def feature_cell(feature, supports, box, reach):
    spec = feature["face_constraints"]["feature_space"]
    if spec["construction"] == "minkowski_extrusion":
        return boolean(BRepAlgoAPI_Common,box,minkowski_cell(spec,supports,reach))
    result = box
    for node in spec["boundary_nodes"]:
        support = supports[node]
        if support["type"] == "plane":
            result = keep_halfspace(result,support["normal"],support["offset"])
        else:
            op = BRepAlgoAPI_Common if support["sigma"] < 0 else BRepAlgoAPI_Cut
            result = boolean(op,result,cylinder(support,reach))
    return result


def shape_meshes(shape, label, red=False, omit_display_caps=False):
    BRepMesh_IncrementalMesh(shape,0.007,False,0.12,True).Perform()
    result = []
    explorer = TopExp_Explorer(shape,TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        surface = BRepAdaptor_Surface(face)
        if omit_display_caps and surface.GetType() == GeomAbs_Plane:
            plane_ = surface.Plane()
            n = np.array(plane_.Axis().Direction().Coord())
            p = np.array(plane_.Location().Coord())
            if any(abs(abs(n[i])-1)<1e-7 and abs(abs(p[i])-LIMIT)<1e-7 for i in range(3)):
                continue
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face,location)
        if triangulation is None:
            continue
        vertices = [triangulation.Node(i).Transformed(location.Transformation()).Coord()
                    for i in range(1,triangulation.NbNodes()+1)]
        triangles = []
        for i in range(1,triangulation.NbTriangles()+1):
            t = [j-1 for j in triangulation.Triangle(i).Get()]
            triangles.append(t[::-1] if face.Orientation() == TopAbs_REVERSED else t)
        if triangles:
            result.append(mesh(vertices,triangles,label,red))
    return result


def axes_traces(frame):
    traces = []
    # Relocated direction glyph only. Bbox coordinates still use model (0,0,0).
    origin = np.array([1.03,-1.03,-.65])
    for label,axis,color in zip("xyz",frame.T,["#e63946","#267a36","#245ddd"]):
        tip = origin+.35*axis
        traces.append(go.Scatter3d(x=[origin[0],tip[0]],y=[origin[1],tip[1]],z=[origin[2],tip[2]],
            mode="lines+text",text=["",f"+{label}"],textposition="top center",
            line=dict(color=color,width=7),showlegend=False,hoverinfo="skip"))
        traces.append(go.Cone(x=[tip[0]],y=[tip[1]],z=[tip[2]],u=[axis[0]],v=[axis[1]],w=[axis[2]],
            sizemode="absolute",sizeref=.09,anchor="tip",showscale=False,
            colorscale=[[0,color],[1,color]],hoverinfo="skip"))
    traces.append(go.Scatter3d(x=[origin[0]],y=[origin[1]],z=[origin[2]],mode="markers+text",
        text=["CRS directions (relocated)"],textposition="bottom center",marker=dict(size=3,color="#222"),
        showlegend=False,hoverinfo="skip"))
    return traces


def trim_warnings(feature, supports):
    warnings = []
    for rule in feature["face_constraints"].get("trim_constraints",[]):
        s = supports[rule["node"]]
        angles = np.unwrap(np.arctan2(s["path"][:,1]-s["center"][1],
                                     s["path"][:,0]-s["center"][0]))
        # Viewer reference patches are single contiguous arcs; extraction must
        # merge periodic interval unions for fragmented real-world patches.
        span = float(np.ptp(angles))
        bound = rule["radians"]
        valid = {"<":span < bound-1e-9, "<=":span <= bound+1e-9,
                 ">":span > bound+1e-9}[rule["operator"]]
        if not valid:
            warnings.append(f"S{rule['node']} trim {math.degrees(span):.1f}° violates {rule['operator']} {math.degrees(bound):g}°")
    return warnings


@lru_cache(maxsize=60)
def geometry_layers(name, ratio):
    feature = FEATURES[name]
    layers = build_layers(feature)
    supports = layers["supports"]
    frame = local_frame(feature,supports)
    points = face_points(layers,frame)
    local = points @ frame
    low, high = local.min(axis=0),local.max(axis=0)
    delta = ratio*(high-low)
    if np.any(high-low <= 1e-9):
        raise ValueError("Degenerate local face envelope")
    box = oriented_box(frame,low-delta,high+delta)
    display_box = BRepPrimAPI_MakeBox(gp_Pnt(-LIMIT,-LIMIT,-LIMIT),2*LIMIT,2*LIMIT,2*LIMIT).Shape()
    reach = 10*max(LIMIT,float(np.max(np.abs(points)))+float(np.max(delta)))
    display = feature_cell(feature,supports,display_box,reach)
    domain = feature_cell(feature,supports,box,reach)
    layers["feature_space"] = shape_meshes(display,"Feature space (including Minkowski continuations)",omit_display_caps=True)
    layers["bbox"] = shape_meshes(box,f"Local-CRS bbox · {ratio:.0%} per side")
    for trace in layers["bbox"]:
        trace.update(color="#407bc4",opacity=.16)
    layers["finite_collision_domain"] = shape_meshes(domain,"Finite collision domain",True)
    for trace in layers["finite_collision_domain"]:
        trace.update(opacity=.3)
    layers["axes"] = axes_traces(frame)
    layers["warnings"] = trim_warnings(feature,supports)
    return layers


def sag_figure(feature):
    nodes=feature["nodes"]
    count=len(nodes)
    pos={n["id"]:np.array([math.cos(math.pi/2+2*math.pi*i/count),math.sin(math.pi/2+2*math.pi*i/count)]) for i,n in enumerate(nodes)}
    fig=go.Figure()
    for index,edge in enumerate(feature["edges"]):
        a,b=(pos[i] for i in edge["nodes"])
        fig.add_trace(go.Scatter(x=[a[0],b[0]],y=[a[1],b[1]],mode="lines",line=dict(color="#8493a5",width=1.5),hovertemplate=edge["code"]+"<extra></extra>",showlegend=False))
        # Stagger labels along complete-graph chords to avoid central overlap.
        fraction=.28+.11*(index%4)
        p=a+(b-a)*fraction
        fig.add_annotation(x=p[0],y=p[1],text=edge["code"],showarrow=False,bgcolor="white",borderpad=2,font=dict(size=12,color="#173149",family="monospace"))
    for node in nodes:
        p=pos[node["id"]]
        fig.add_trace(go.Scatter(x=[p[0]],y=[p[1]],mode="markers+text",text=[f"S{node['id']}"],textposition="middle center",
                                marker=dict(size=39,color="#b9e6cc" if node["surface_type"]=="plane" else "#a5d0f5",line=dict(color="#234",width=1)),
                                hovertemplate=f"{node['role']}<br>{node['surface_type']}<extra></extra>",showlegend=False))
        fig.add_annotation(x=p[0]*1.24,y=p[1]*1.24,text=node["role"],showarrow=False,font=dict(size=11))
    fig.update_layout(title="Surface Attributed Graph",template="plotly_white",margin=dict(l=25,r=25,t=55,b=25),dragmode="pan",
                      xaxis=dict(visible=False,range=[-1.65,1.65]),yaxis=dict(visible=False,range=[-1.65,1.65],scaleanchor="x"),uirevision=feature["name"])
    return fig


app=Dash(__name__)
app.title="Feature-space Seed Definition"
app.layout=html.Div([
    html.Div([
        html.H4("Feature-space Seed Definition",style={"margin":"0 15px 0 0"}),
        dcc.Dropdown(id="feature",options=[{"label":n,"value":n} for n in FEATURES],value=next(iter(FEATURES)),clearable=False,style={"width":"300px"}),
        html.Div([
            dcc.Checklist(id="layers",options=[
                {"label":" Supporting surfaces  ","value":"surfaces"},
                {"label":" Feature space  ","value":"feature_space"},
                {"label":" Topological faces  ","value":"trimmed"},
                {"label":" Bbox  ","value":"bbox"},
                {"label":" Finite collision domain  ","value":"finite_collision_domain"}
            ],value=["surfaces"],inline=True),
            html.Div([
                html.Span("Bbox extension ratio (each side)",style={"fontSize":"12px"}),
                dcc.Slider(id="extension-ratio",min=.1,max=.3,step=.1,
                    marks={.1:"0.1",.2:"0.2",.3:"0.3"},value=.1,
                    tooltip={"placement":"bottom","always_visible":False})
            ],style={"maxWidth":"400px","paddingTop":"5px"})
        ],style={"marginLeft":"20px","flex":"1","minWidth":"350px"})
    ],style={"display":"flex","alignItems":"center","padding":"8px 12px","background":"#f5f6f7","flexWrap":"wrap"}),
    html.Div(id="description",style={"padding":"6px 14px","fontSize":"13px"}),
    html.Div([
        dcc.Graph(id="viewer",style={"height":"86vh","minWidth":0},config={"displaylogo":False,"scrollZoom":True}),
        dcc.Graph(id="sag",style={"height":"86vh","minWidth":0},config={"displaylogo":False,"scrollZoom":True})
    ],style={"display":"grid","gridTemplateColumns":"minmax(0, 3fr) minmax(0, 2fr)"})
])


@app.callback(
    Output("viewer","figure"),
    Output("sag","figure"),
    Output("description","children"),
    Input("feature","value"),
    Input("layers","value"),
    Input("extension-ratio","value")
)
def update_view(name, visible, ratio=.1):
    feature = FEATURES[name]
    layers = geometry_layers(name,float(ratio))

    visible = visible or []

    traces = [
        trace
        for layer in visible
        for trace in layers[layer]
    ]

    # Retain the original topological patches and their gold trim boundaries.
    if "trimmed" in visible:
        traces.extend(layers["boundaries"])
    traces.extend(layers["axes"])

    fig = go.Figure(traces)

    fig.update_layout(
        template="plotly_white",
        uirevision="seed-camera",
        margin=dict(l=0,r=0,t=10,b=0),
        showlegend=False,
        scene=dict(
            uirevision="seed-camera",
            dragmode="orbit",
            aspectmode="cube",
            xaxis=dict(range=[-LIMIT,LIMIT]),
            yaxis=dict(range=[-LIMIT,LIMIT]),
            zaxis=dict(range=[-LIMIT,LIMIT])
        )
    )

    return (
        fig,
        sag_figure(feature),
        f"{name} · {feature['space_model']} · "
        f"{len(feature['nodes'])} surfaces · "
        f"Local CRS origin = model origin; each bbox side expands by {ratio:.0%}; "
        "finite collision domain = feature space ∩ expanded local bbox. "
        "Viewer templates only; no part-solid collision test. "
        + ("⚠ Preserved template: " + "; ".join(layers["warnings"]) if layers["warnings"] else "")
    )

if __name__=="__main__":
    app.run(host=HOST,port=PORT,debug=False)
