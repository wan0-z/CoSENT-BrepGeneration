from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

HERE = Path(__file__).resolve().parent
SEED_PATH = HERE.parent / "Seed_Extractor" / "data" / "feature_space_seeds.json"
FEATURES = {f["name"]: f for f in json.loads(SEED_PATH.read_text())["features"]}
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
    layers = {"surfaces":[], "trimmed":[], "boundaries":[], "recession":[]}
    cube = box_planes([-LIMIT]*3,[LIMIT]*3)
    bounds = box_planes(lo,hi)
    for i,(n,d) in enumerate(planes):
        label = f"S{feature['nodes'][i]['id']} · {feature['nodes'][i]['role']}"
        layers["surfaces"].append(polygon_mesh(plane_polygon(n,d,cube),label))
        poly = plane_polygon(n,d,bounds+planes)
        if len(poly) < 3:
            raise ValueError(f"Inactive boundary: {feature['name']} {label}")
        layers["trimmed"].append(polygon_mesh(poly,label))
        layers["boundaries"].append(boundary_line(poly, label, close=True))
    ratio = feature["face_constraints"]["recession_volume"].get("extension_ratio",.1)
    delta = ratio*(hi-lo)
    elo = lo - delta*np.array([o[0] for o in opens])
    ehi = hi + delta*np.array([o[1] for o in opens])
    constraints = planes+box_planes(elo,ehi)
    seen = set()
    for n,d in constraints:
        key = tuple(np.round([*n,d],8))
        if key in seen:
            continue
        seen.add(key)
        poly = plane_polygon(n,d,constraints)
        if polygon_area(poly)>1e-9:
            layers["recession"].append(polygon_mesh(poly,"Collision boundary",True))
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
    ratio=feature["face_constraints"]["recession_volume"].get("extension_ratio",.1)
    layers = {"surfaces":[], "trimmed":[],"boundaries": [], "recession":[]}
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
        extended=contour
    elif name == "v_circular_end_blind_slot":
        paths=[np.array([[c,r],[-c,r]]),arc(-c,0,r,math.pi/2,3*math.pi/2),
               np.array([[-c,-r],[c,-r]])]
        cylinders={1:(-c,0,r)}
        contour=np.concatenate(paths)
        extended=contour.copy()
        extended[np.isclose(extended[:,0],c),0]+=ratio*(2*c+r)
    elif name == "h_circular_end_blind_slot":
        paths=[arc(-c,0,r,math.pi,1.5*math.pi),np.array([[-c,-r],[c,-r]]),
               arc(c,0,r,1.5*math.pi,2*math.pi)]
        cylinders={0:(-c,0,r),2:(c,0,r)}
        contour=np.concatenate(paths)
        # Tangent continuation of both open arc endpoints, +10% of height.
        extended=np.vstack([[contour[0,0],ratio*r],contour,[contour[-1,0],ratio*r]])
    else:
        r=.62
        partial=name in {"circular_through_slot","circular_blind_step"}
        path=arc(0,0,r,0,math.pi if partial else 2*math.pi)
        paths=[path]
        cylinders={0:(0,0,r)}
        contour=arc(0,0,r,0,2*math.pi)
        extended=contour
        if name == "circular_blind_step":
            contour=path
            extended=np.vstack([[r,-ratio*r],path,[-r,-ratio*r]])
    ring=name=="Oring"
    if ring:
        paths=[arc(0,0,.62,0,2*math.pi),None,arc(0,0,.30,0,2*math.pi)]
        cylinders={0:(0,0,.62),2:(0,0,.30)}
    nodes=feature["nodes"]
    for i,node in enumerate(nodes):
        label=f"S{node['id']} · {node['role']}"
        if node["surface_type"]=="cylinder":
            cx, cy, rad = cylinders[i]

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
    low=z0 if blind else z0-ratio*(z1-z0)
    high=z1+ratio*(z1-z0)
    if ring:
        layers["recession"]=[wall(arc(0,0,.62,0,2*math.pi),low,high,"Outer cylinder",True),
                             wall(arc(0,0,.30,0,2*math.pi),low,high,"Inner cylinder",True),
                             annular_cap(.62,.30,low,"Bottom",True),annular_cap(.62,.30,high,"Open-end cap",True)]
    else:
        closed=np.vstack([extended,extended[0]])
        layers["recession"]=[wall(closed,low,high,"Collision side boundary",True),
                             polygon_mesh([[x,y,low] for x,y in extended],"Lower cap",True),
                             polygon_mesh([[x,y,high] for x,y in extended],"Upper cap",True)]
    return layers


def build_layers(feature):
    return planar_layers(feature) if planar_spec(feature["name"]) is not None else curved_layers(feature)


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
        dcc.Checklist(id="layers",options=[{"label":" Supporting surfaces  ","value":"surfaces"},{"label":" Trimmed faces  ","value":"trimmed"},{"label":" Collision volume (10%)","value":"recession"}],value=["surfaces"],inline=True,style={"marginLeft":"20px"})
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
    Input("layers","value")
)
def update_view(name, visible):
    feature = FEATURES[name]
    layers = build_layers(feature)

    visible = visible or []

    traces = [
        trace
        for layer in visible
        for trace in layers[layer]
    ]

    # Trim boundaries are part of the trimmed-face visualization.
    if "trimmed" in visible:
        traces.extend(layers["boundaries"])

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
        "Finite collision volume extends only through open boundaries by 10%."
    )

if __name__=="__main__":
    app.run(host=HOST,port=PORT,debug=False)
