# -*- coding: utf-8 -*-
import argparse
from multiprocessing.pool import Pool
import gc
import json
import os.path as osp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import repeat

from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend import TopologyUtils
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_FORWARD, TopAbs_REVERSED 
from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID)
from OCC.Core.TopoDS import (
    TopoDS_Solid,
    TopoDS_Compound,
    TopoDS_CompSolid,
)
from OCC.Core.TopExp import topexp
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle, 
                              GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola, 
                              GeomAbs_BezierCurve, GeomAbs_BSplineCurve, 
                              GeomAbs_OffsetCurve, GeomAbs_OtherCurve)
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties

# occwl
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.uvgrid import uvgrid
from occwl.graph import face_adjacency

from types import SimpleNamespace



def scale_solid_to_unit_box(solid):
    if isinstance(solid, Solid):
        return solid.scale_to_unit_box(copy=True)
    solid = Solid(solid, allow_compound=True)
    solid = solid.scale_to_unit_box(copy=True)
    return solid.topods_shape()


class TopologyChecker():
    # modified from BREPNET: https://github.com/AutodeskAILab/BRepNet/blob/master/pipeline/extract_brepnet_data_from_step.py
    def __init__(self):
        pass

    def find_edges_from_wires(self, top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set

    def find_edges_from_top_exp(self, top_exp):
        edge_set = set(top_exp.edges())
        return edge_set

    def check_closed(self, body):
        # In Open Cascade, unlinked (open) edges can be identified
        # as they appear in the edges iterator when ignore_orientation=False
        # but are not present in any wire
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0

    def check_manifold(self, top_exp):
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False
                faces.add(face)
        return True

    def check_unique_coedges(self, top_exp):
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                # We want to detect the case where the coedges
                # are not unique
                if tup in coedge_set:
                    return False
                coedge_set.add(tup)
        return True

    def __call__(self, body):
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        if top_exp.number_of_faces() == 0:
            print('Empty shape') 
            return False
        # OCC.BRepCheck, perform topology and geometricals check
        analyzer = BRepCheck_Analyzer(body)
        if not analyzer.IsValid(body):
            print('BRepCheck_Analyzer found defects') 
            return False
        # other topology check
        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return False
        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return False
        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is uses in multiple loops are not supported")
            return False
        return True


class AAGExtractor:
    def __init__(
        self, 
        step_file, 
        attribute_schema, 
        scale_body=True):
        self.step_file = step_file
        self.attribute_schema = attribute_schema
        self.scale_body = scale_body
        # whether to extract UV-grid
        self.use_uv = "UV-grid" in self.attribute_schema.keys()
        self.topchecker = TopologyChecker()
        if self.use_uv:
            # UV-gird size for face
            self.num_srf_u = self.attribute_schema["UV-grid"]["num_srf_u"]
            self.num_srf_v = self.attribute_schema["UV-grid"]["num_srf_v"]
            # UV-gird size for curve
            self.num_crv_u = self.attribute_schema["UV-grid"]["num_crv_u"]

    def process(self):
        """
        Creates a attributed adjacency graph from the given shape (Solid)

        Args:
            
        Returns:

        """
        # Load the body from the STEP file
        self.body = self.load_body_from_step()
        assert self.body is not None, \
            "the shape {} is non-manifold or open".format(self.step_file)
        assert self.topchecker(self.body), \
            "the shape {} has wrong topology".format(self.step_file)
        assert isinstance(self.body, TopoDS_Solid), \
            'file {} is {}, not TopoDS_Solid'.format(self.step_file, type(self.body))
        
        # We want to apply a transform so that the solid
        # is centered on the origin and scaled so it just fits
        # into a box [-1, 1]^3
        if self.scale_body:
            self.body = scale_solid_to_unit_box(self.body)

        # Build face adjacency graph with B-rep entities as node and edge features
        # occwl.Solid is a occwl warp for TopoDS_Shape
        try:
            graph = face_adjacency(Solid(self.body))
        except Exception as e:
            print(e)
            assert False, 'Wrong shape {}'.format(self.step_file)

        # get the attributes for faces
        graph_face_attr = []
        graph_face_grid = []
        # the FaceCentroidAttribute has xyz coordinate
        # so the length of face attributes should add 2 if containing centroid
        len_of_face_attr = len(self.attribute_schema["face_attributes"]) + \
            2 if "FaceCentroidAttribute" in self.attribute_schema["face_attributes"] else 0
        for face_idx in graph.nodes:
            # Get the B-rep face
            face = graph.nodes[face_idx]["face"]
            # get the attributes from face
            face_attr = self.extract_attributes_from_face(
                face.topods_shape())  # from occwl.Face to OCC.TopoDS_Face
            assert len_of_face_attr == len(face_attr)
            graph_face_attr.append(face_attr)
            # get the UV point grid from face
            if self.use_uv and self.num_srf_u and self.num_srf_v:
                uv_grid = self.extract_face_point_grid(face)
                assert uv_grid.shape[0] == 7
                graph_face_grid.append(uv_grid.tolist())

        graph_edge_attr = []
        graph_edge_grid = []
        for edge_idx in graph.edges:
            edge = graph.edges[edge_idx]["edge"]
            # Ignore dgenerate edges, e.g. at apex of cone
            if not edge.has_curve():
                continue
            # get the attributes from edge
            edge = edge.topods_shape() # from occwl.Edge to OCC.TopoDS_Edge
            edge_attr = self.extract_attributes_from_edge(edge)
            assert len(self.attribute_schema["edge_attributes"]) == len(edge_attr)
            graph_edge_attr.append(edge_attr)
            # get the UV point grid from edge
            if self.use_uv and self.num_crv_u:
                u_grid = self.extract_edge_point_grid(edge)
                assert u_grid.shape[0] == 12
                graph_edge_grid.append(u_grid.tolist())
        
        # get graph from nx.DiGraph
        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        graph = { 
            'edges': (src, dst),
            'num_nodes': len(graph.nodes)
        }

        return { 
            'graph': graph,
            'graph_face_attr': graph_face_attr,
            'graph_face_grid': graph_face_grid,
            'graph_edge_attr': graph_edge_attr,
            'graph_edge_grid': graph_edge_grid,
        }

    ########################
    # Step Loader
    ########################

    def load_body_from_step(self):
        """
        Load the body from the step file.  
        We expect only one body in each file
        """
        step_filename_str = str(self.step_file)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename_str)
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape

    ########################
    # Face Attributes Extractor
    ########################

    def extract_attributes_from_face(self, face) -> list:
        def plane_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Plane:
                return 1.0
            return 0.0

        def cylinder_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Cylinder:
                return 1.0
            return 0.0

        def cone_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Cone:
                return 1.0
            return 0.0

        def sphere_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Sphere:
                return 1.0
            return 0.0

        def torus_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Torus:
                return 1.0
            return 0.0

        def area_attribute(face):
            geometry_properties = GProp_GProps()
            brepgprop_SurfaceProperties(face, geometry_properties)
            return geometry_properties.Mass()

        def rational_nurbs_attribute(face):
            surf = BRepAdaptor_Surface(face)
            if surf.GetType() == GeomAbs_BSplineSurface:
                bspline = surf.BSpline()
            elif surf.GetType() == GeomAbs_BezierSurface:
                bspline = surf.Bezier()
            else:
                bspline = None
            
            if bspline is not None:
                if bspline.IsURational() or bspline.IsVRational():
                    return 1.0
            return 0.0

        def centroid_attribute(face):
            mass_props = GProp_GProps()
            brepgprop_SurfaceProperties(face, mass_props)
            gPt = mass_props.CentreOfMass()

            return gPt.Coord()

        face_attributes = []
        for attribute in self.attribute_schema["face_attributes"]:
            if attribute == "Plane":
                face_attributes.append(plane_attribute(face))
            elif attribute == "Cylinder":
                face_attributes.append(cylinder_attribute(face))
            elif attribute == "Cone":
                face_attributes.append(cone_attribute(face))
            elif attribute == "SphereFaceAttribute":
                face_attributes.append(sphere_attribute(face))
            elif attribute == "TorusFaceAttribute":
                face_attributes.append(torus_attribute(face))
            elif attribute == "FaceAreaAttribute":
                face_attributes.append(area_attribute(face))
            elif attribute == "RationalNurbsFaceAttribute":
                face_attributes.append(rational_nurbs_attribute(face))
            elif attribute == "FaceCentroidAttribute":
                face_attributes.extend(centroid_attribute(face))
            else:
                assert False, "Unknown face attribute"
        return face_attributes
        
    ########################
    # Edge attributes Extractor
    ########################

    def extract_attributes_from_edge(self, edge) -> list:
        def find_edge_convexity(edge, faces):
            edge_data = EdgeDataExtractor(Edge(edge), 
                faces, use_arclength_params=False)
            if not edge_data.good:
                # This is the case where the edge is a pole of a sphere
                return 0.0
            angle_tol_rads = 0.0872664626 # 5 degrees 
            convexity = edge_data.edge_convexity(angle_tol_rads)
            return convexity

        def convexity_attribute(convexity, attribute):
            if attribute == "Convex edge":
                return convexity == EdgeConvexity.CONVEX
            if attribute == "Concave edge":
                return convexity == EdgeConvexity.CONCAVE
            if attribute == "Smooth":
                return convexity == EdgeConvexity.SMOOTH
            assert False, "Unknown convexity"
            return 0.0

        def edge_length_attribute(edge):
            geometry_properties = GProp_GProps()
            brepgprop_LinearProperties(edge, geometry_properties)
            return geometry_properties.Mass()

        def circular_edge_attribute(edge):
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            if curv_type == GeomAbs_Circle:
                return 1.0
            return 0.0

        def closed_edge_attribute(edge):
            if BRep_Tool().IsClosed(edge):
                return 1.0
            return 0.0

        def elliptical_edge_attribute(edge):
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            if curv_type == GeomAbs_Ellipse:
                return 1.0
            return 0.0

        def helical_edge_attribute(edge):
            # We don't have this attribute in Open Cascade
            assert False, "Not implemented for the OCC pipeline"
            return 0.0

        def int_curve_edge_attribute(edge):
            # We don't have this attribute in Open Cascade
            assert False, "Not implemented for the OCC pipeline"
            return 0.0

        def straight_edge_attribute(edge):
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            if curv_type == GeomAbs_Line:
                return 1.0
            return 0.0

        def hyperbolic_edge_attribute(edge):
            if Edge(edge).curve_type() == "hyperbola":
                return 1.0
            return 0.0

        def parabolic_edge_attribute(edge):
            if Edge(edge).curve_type() == "parabola":
                return 1.0
            return 0.0

        def bezier_edge_attribute(edge):
            if Edge(edge).curve_type() == "bezier":
                return 1.0
            return 0.0

        def non_rational_bspline_edge_attribute(edge):
            occwl_edge = Edge(edge)
            if occwl_edge.curve_type() == "bspline" and not occwl_edge.rational():
                return 1.0
            return 0.0

        def rational_bspline_edge_attribute(edge):
            occwl_edge = Edge(edge)
            if occwl_edge.curve_type() == "bspline" and occwl_edge.rational():
                return 1.0
            return 0.0

        def offset_edge_attribute(edge):
            if Edge(edge).curve_type() == "offset":
                return 1.0
            return 0.0
        
        # get the faces from an edge
        top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]

        attribute_list = self.attribute_schema["edge_attributes"]
        if "Concave edge" in attribute_list or \
            "Convex edge" in attribute_list or \
            "Smooth"  in attribute_list:
            convexity = find_edge_convexity(edge, faces_of_edge)
        edge_attributes = []
        for attribute in attribute_list:
            if attribute == "Concave edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Convex edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Smooth":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "EdgeLengthAttribute":
                edge_attributes.append(edge_length_attribute(edge))
            elif attribute == "CircularEdgeAttribute":
                edge_attributes.append(circular_edge_attribute(edge))
            elif attribute == "ClosedEdgeAttribute":
                edge_attributes.append(closed_edge_attribute(edge))
            elif attribute == "EllipticalEdgeAttribute":
                edge_attributes.append(elliptical_edge_attribute(edge))
            elif attribute == "HelicalEdgeAttribute":
                edge_attributes.append(helical_edge_attribute(edge))
            elif attribute == "IntcurveEdgeAttribute":
                edge_attributes.append(int_curve_edge_attribute(edge))
            elif attribute == "StraightEdgeAttribute":
                edge_attributes.append(straight_edge_attribute(edge))
            elif attribute == "HyperbolicEdgeAttribute":
                edge_attributes.append(hyperbolic_edge_attribute(edge))
            elif attribute == "ParabolicEdgeAttribute":
                edge_attributes.append(parabolic_edge_attribute(edge))
            elif attribute == "BezierEdgeAttribute":
                edge_attributes.append(bezier_edge_attribute(edge))
            elif attribute == "NonRationalBSplineEdgeAttribute":
                edge_attributes.append(non_rational_bspline_edge_attribute(edge))
            elif attribute == "RationalBSplineEdgeAttribute":
                edge_attributes.append(rational_bspline_edge_attribute(edge))
            elif attribute == "OffsetEdgeAttribute":
                edge_attributes.append(offset_edge_attribute(edge))
            else:
                assert False, "Unknown face attribute"
        return edge_attributes

    ########################
    # Face UV Point Grid Extractor
    ########################

    def extract_face_point_grid(self, face) -> np.array:
        """
        Extract a UV-Net point grid from the given face.

        Returns a tensor [ 7 x num_pts_u x num_pts_v ]

        For each point the values are 
        
            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)
            - Trimming mast

        """
        points = uvgrid(face, self.num_srf_u, self.num_srf_v, method="point")
        normals = uvgrid(face, self.num_srf_u, self.num_srf_v, method="normal")
        mask = uvgrid(face, self.num_srf_u, self.num_srf_v, method="inside")

        # This has shape [ num_pts_u x num_pts_v x 7 ]
        single_grid = np.concatenate([points, normals, mask], axis=2)
       
        return np.transpose(single_grid, (2, 0, 1))

    ########################
    # Edge UV Point Grid Extractor
    ########################

    def extract_edge_point_grid(self, edge) -> np.array:
        """
        Extract a edge grid (aligned with the coedge direction).

        The edge grids will be of size

            [ 12 x num_u ]

        The values are

            - x, y, z    (coordinates of the points)
            - tx, ty, tz (tangent of the curve, oriented to match the coedge)
            - Lx, Ly, Lz (Normal for the left face)
            - Rx, Ry, Rz (Normal for the right face)
        """

        # get the faces from an edge
        top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]

        edge_data = EdgeDataExtractor(Edge(edge), faces_of_edge, 
            num_samples=self.num_crv_u, use_arclength_params=True)
        if not edge_data.good:
            # We hit a problem evaluating the edge data.  This may happen if we have
            # an edge with not geometry (like the pole of a sphere).
            # In this case we return zeros
            return np.zeros((12, self.num_crv_u)) 

        single_grid = np.concatenate(
            [
                edge_data.points, 
                edge_data.tangents, 
                edge_data.left_normals,
                edge_data.right_normals
            ],
            axis = 1
        )
        return np.transpose(single_grid, (1, 0))


#######################################
# Calculate mean & std of attributes
#######################################

def check_zero_std(stat_data):
    std_face_attr = stat_data['std_face_attr']
    std_edge_attr = stat_data['std_edge_attr']
    if np.nonzero(std_face_attr)[0].shape[0] != len(std_face_attr) \
        or np.nonzero(std_edge_attr)[0].shape[0] != len(std_edge_attr):
        print('WARNING! has zero standard deviation.')


def find_standardization(data):
    '''
    Find mean and standard deviation of face and edge attributes
    Args:
    data (list): [filename, graph_data]
    '''
    all_face_attr = []
    all_edge_attr = []
    for one_sample in data:
        fn, graph = one_sample
        all_face_attr.extend(graph["graph_face_attr"])
        all_edge_attr.extend(graph["graph_edge_attr"])
    graph_face_attr = np.asarray(all_face_attr)
    graph_edge_attr = np.asarray(all_edge_attr)

    mean_face_attr = np.mean(graph_face_attr, axis=0)
    std_face_attr = np.std(graph_face_attr, axis=0)

    mean_edge_attr = np.mean(graph_edge_attr, axis=0)
    std_edge_attr = np.std(graph_edge_attr, axis=0)

    return {
            'mean_face_attr': mean_face_attr.tolist(),
            'std_face_attr': std_face_attr.tolist(),
            'mean_edge_attr': mean_edge_attr.tolist(),
            'std_edge_attr': std_edge_attr.tolist(),
        }

########################
# Save all step files in a json
########################

def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def save_json_data(pathname, data):
    """Export a data to a json file"""
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def initializer():
    import signal
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def find_step_file(steps_dir, sample_name):
    """
    Find corresponding STEP file for one sample.

    Supports:
        sample.step
        sample.stp
        sample.STEP
        sample.STP
    """
    candidates = [
        steps_dir / f"{sample_name}.step",
        steps_dir / f"{sample_name}.stp",
        steps_dir / f"{sample_name}.STEP",
        steps_dir / f"{sample_name}.STP",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def load_sample_names_from_labels(labels_dir, require_rel_label=False):
    """
    Load sample names from labels folder.

    It uses labels/{sample}.json as main label file.
    It ignores labels/{sample}_rel.json.

    If require_rel_label=True, it only keeps samples that also have:
        labels/{sample}_rel.json
    """
    labels_dir = Path(labels_dir)

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels folder not found: {labels_dir}")

    sample_names = []

    for label_path in labels_dir.glob("*.json"):
        name = label_path.stem

        if name.endswith("_rel"):
            continue

        if require_rel_label:
            rel_label_path = labels_dir / f"{name}_rel.json"
            if not rel_label_path.exists():
                continue

        sample_names.append(name)

    sample_names = sorted(sample_names, key=extract_sample_index)

    return sample_names


def extract_sample_index(sample_name):
    """
    Example:
        20240116_231044_123_result -> 123

    If extraction fails, return a very large number.
    """
    try:
        return int(sample_name.split("_")[-2])
    except Exception:
        return 10**18


def save_single_graph_json(pathname, sample_name, graph_data, save_format="pair"):
    """
    Save one graph sample.

    save_format:
        "pair":
            Save in the same item format as original graphs.json:
                [sample_name, graph_data]

        "graph":
            Save only graph_data:
                {
                    "graph": ...,
                    "graph_face_attr": ...,
                    ...
                }

        "dict":
            Save a dict with filename:
                {
                    "filename": sample_name,
                    "graph_data": graph_data
                }
    """
    pathname = Path(pathname)

    if save_format == "pair":
        data_to_save = [sample_name, graph_data]
    elif save_format == "graph":
        data_to_save = graph_data
    elif save_format == "dict":
        data_to_save = {
            "filename": sample_name,
            "graph_data": graph_data,
        }
    else:
        raise ValueError(
            f"Unknown save_format: {save_format}. "
            "Use 'pair', 'graph', or 'dict'."
        )

    with open(pathname, "w", encoding="utf8") as fp:
        json.dump(data_to_save, fp, indent=4, ensure_ascii=False, sort_keys=False)


def process_one_file_to_single_json(args):
    """
    Process one STEP file and save one graph JSON file.

    This function is used by multiprocessing workers.
    """
    sample_name, step_file, output_path, feature_schema, save_format, overwrite = args

    output_path = Path(output_path)
    out_file = output_path / f"{sample_name}.json"

    if out_file.exists() and not overwrite:
        return {
            "sample_name": sample_name,
            "status": "skipped_exists",
            "num_faces": 0,
            "num_edges": 0,
            "face_attr_sum": None,
            "face_attr_sumsq": None,
            "edge_attr_sum": None,
            "edge_attr_sumsq": None,
        }

    extractor = AAGExtractor(step_file, feature_schema)
    graph_data = extractor.process()

    save_single_graph_json(
        pathname=out_file,
        sample_name=sample_name,
        graph_data=graph_data,
        save_format=save_format,
    )

    face_attr = np.asarray(graph_data["graph_face_attr"], dtype=np.float64)
    edge_attr = np.asarray(graph_data["graph_edge_attr"], dtype=np.float64)

    result = {
        "sample_name": sample_name,
        "status": "success",
        "num_faces": int(face_attr.shape[0]),
        "num_edges": int(edge_attr.shape[0]),
        "face_attr_sum": face_attr.sum(axis=0).tolist() if face_attr.size > 0 else None,
        "face_attr_sumsq": np.square(face_attr).sum(axis=0).tolist() if face_attr.size > 0 else None,
        "edge_attr_sum": edge_attr.sum(axis=0).tolist() if edge_attr.size > 0 else None,
        "edge_attr_sumsq": np.square(edge_attr).sum(axis=0).tolist() if edge_attr.size > 0 else None,
    }

    return result


def update_running_stat(running, one_result):
    """
    Update global mean/std statistics without storing all graph data in memory.
    """
    if one_result["status"] != "success":
        return

    num_faces = one_result["num_faces"]
    num_edges = one_result["num_edges"]

    if num_faces > 0 and one_result["face_attr_sum"] is not None:
        face_sum = np.asarray(one_result["face_attr_sum"], dtype=np.float64)
        face_sumsq = np.asarray(one_result["face_attr_sumsq"], dtype=np.float64)

        if running["face_sum"] is None:
            running["face_sum"] = face_sum
            running["face_sumsq"] = face_sumsq
        else:
            running["face_sum"] += face_sum
            running["face_sumsq"] += face_sumsq

        running["face_count"] += num_faces

    if num_edges > 0 and one_result["edge_attr_sum"] is not None:
        edge_sum = np.asarray(one_result["edge_attr_sum"], dtype=np.float64)
        edge_sumsq = np.asarray(one_result["edge_attr_sumsq"], dtype=np.float64)

        if running["edge_sum"] is None:
            running["edge_sum"] = edge_sum
            running["edge_sumsq"] = edge_sumsq
        else:
            running["edge_sum"] += edge_sum
            running["edge_sumsq"] += edge_sumsq

        running["edge_count"] += num_edges


def finalize_running_stat(running):
    """
    Create attr_stat.json data from streaming statistics.
    """
    if running["face_count"] == 0:
        raise RuntimeError("No face attributes were collected.")

    if running["edge_count"] == 0:
        raise RuntimeError("No edge attributes were collected.")

    mean_face_attr = running["face_sum"] / running["face_count"]
    mean_edge_attr = running["edge_sum"] / running["edge_count"]

    var_face_attr = running["face_sumsq"] / running["face_count"] - np.square(mean_face_attr)
    var_edge_attr = running["edge_sumsq"] / running["edge_count"] - np.square(mean_edge_attr)

    var_face_attr = np.maximum(var_face_attr, 0.0)
    var_edge_attr = np.maximum(var_edge_attr, 0.0)

    std_face_attr = np.sqrt(var_face_attr)
    std_edge_attr = np.sqrt(var_edge_attr)

    attr_stat = {
        "mean_face_attr": mean_face_attr.tolist(),
        "std_face_attr": std_face_attr.tolist(),
        "mean_edge_attr": mean_edge_attr.tolist(),
        "std_edge_attr": std_edge_attr.tolist(),
    }

    return attr_stat


def main(args):
    labels_path = Path(args.labels_path)
    step_path = Path(args.step_path)
    output_path = Path(args.output)

    output_path.mkdir(parents=True, exist_ok=True)

    feature_list_path = None
    if args.feature_list is not None:
        feature_list_path = Path(args.feature_list)

    parent_folder = Path(__file__).parent.parent

    if feature_list_path is None:
        feature_list_path = parent_folder / "feature_lists/all.json"

    feature_schema = load_json(feature_list_path)

    sample_names = load_sample_names_from_labels(
        labels_dir=labels_path,
        require_rel_label=args.require_rel_label,
    )

    if args.max_samples is not None:
        sample_names = sample_names[:args.max_samples]

    tasks = []
    missing_steps = []

    for sample_name in sample_names:
        step_file = find_step_file(step_path, sample_name)

        if step_file is None:
            missing_steps.append(sample_name)
            continue

        tasks.append(
            (
                sample_name,
                step_file,
                output_path,
                feature_schema,
                args.save_format,
                args.overwrite,
            )
        )

    print(f"Found label samples: {len(sample_names)}")
    print(f"Found valid STEP tasks: {len(tasks)}")
    print(f"Missing STEP files: {len(missing_steps)}")
    print(f"Output graphs folder: {output_path}")
    print(f"Graph save format: {args.save_format}")

    if missing_steps:
        print("\nFirst 10 missing STEP samples:")
        for name in missing_steps[:10]:
            print(f"  {name}")

    if len(tasks) == 0:
        raise RuntimeError("No valid STEP files found. Please check labels_path and step_path.")

    running = {
        "face_sum": None,
        "face_sumsq": None,
        "face_count": 0,
        "edge_sum": None,
        "edge_sumsq": None,
        "edge_count": 0,
    }

    success_count = 0
    skipped_count = 0
    failed_count = 0

    if args.num_workers <= 1:
        for task in tqdm(tasks, total=len(tasks)):
            try:
                result = process_one_file_to_single_json(task)
                if result["status"] == "success":
                    success_count += 1
                    update_running_stat(running, result)
                elif result["status"] == "skipped_exists":
                    skipped_count += 1
            except Exception as e:
                failed_count += 1
                sample_name = task[0]
                print(f"\nFailed sample: {sample_name}")
                print(f"Error: {e}")
    else:
        pool = Pool(processes=args.num_workers, initializer=initializer)
        try:
            iterator = pool.imap_unordered(process_one_file_to_single_json, tasks)

            for result in tqdm(iterator, total=len(tasks)):
                if result["status"] == "success":
                    success_count += 1
                    update_running_stat(running, result)
                elif result["status"] == "skipped_exists":
                    skipped_count += 1

            pool.close()
            pool.join()

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

        except Exception:
            pool.terminate()
            pool.join()
            raise

    if args.save_attr_stat and success_count > 0:
        attr_stat = finalize_running_stat(running)
        check_zero_std(attr_stat)
        save_json_data(
            output_path / "attr_stat.json",
            attr_stat,
        )

    gc.collect()

    print("\nFinished.")
    print(f"Success: {success_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Graphs saved to: {output_path}")

    if args.save_attr_stat and success_count > 0:
        print(f"Attribute statistics saved to: {output_path / 'attr_stat.json'}")


if __name__ == "__main__":
    args = SimpleNamespace(
        # labels 文件夹位置
        labels_path=r"data\mfinstseg\labels",

        # steps 文件夹位置
        step_path=r"data\mfinstseg\steps",

        # graphs 输出文件夹位置
        output=r"data\mfinstseg\graphs",

        # AAGNet 作者提供的 feature_lists/all.json
        # 建议写绝对路径
        feature_list=r"data\mfinstseg\feature_lists_aag_all.json",

        # 并行进程数。先用 1 测试，没问题后可以改成 4 或 8
        num_workers=1,

        # 最多处理多少个样本；None 表示处理 labels 里的全部样本
        max_samples=None,

        # 是否要求 labels/{sample}_rel.json 存在
        # 你前面设置 require_rel_label=False，所以这里也先设 False
        require_rel_label=False,

        # 如果 graphs/{sample}.json 已经存在，是否覆盖
        overwrite=False,

        # 是否保存 attr_stat.json
        save_attr_stat=True,

        # 每个 graph json 的保存格式
        # 推荐 pair，和原始 graphs.json 中单个 item 的格式一致
        # 可选：'pair', 'graph', 'dict'
        save_format="pair",
    )

    main(args)