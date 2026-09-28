"""Run the unchanged space recognizer and audit GT instances afterwards only.

Usage matches extractor_one_step_space.py. GT is never used by recognition.
"""
import itertools
import json
import math
import extractor_one_step_space as core


def audit(shape, faces, fag, features, cache):
    graph = core.build_oriented_surface_graph(faces, set(range(len(faces))))
    groups = {}
    for face in cache['faces']:
        category = int(face['ground_truth_category_id'])
        if category in (0, 23, 24):
            continue
        category = 8 if category in (10, 20) else category
        groups.setdefault((category, face['ground_truth_instance_id']), set()).add(int(face['face_id']))
    reports = []
    for (category, instance), ids in groups.items():
        feature = next((f for f in features if f['category_id'] == category), None)
        if feature is None:
            continue
        nodes = [n for n, a in graph.nodes(data=True) if ids.intersection(a['face_ids'])]
        candidates = [[n for n in nodes if graph.nodes[n]['surface_type'] == role['surface_type']] for role in feature['nodes']]
        best = None
        for values in itertools.product(*candidates):
            if len(set(values)) != len(values):
                continue
            mapping = {int(role['id']): n for role, n in zip(feature['nodes'], values)}
            mismatches = []
            for edge in feature.get('edges', []):
                a, b = (mapping[int(i)] for i in edge['nodes'])
                actual = graph.edges[a, b]['code'] if graph.has_edge(a, b) else None
                if actual != edge['code']:
                    mismatches.append({'roles': edge['nodes'], 'expected': edge['code'], 'actual': actual})
            trim = [int(role['id']) for role, n in zip(feature['nodes'], values) if n not in core.candidate_nodes(graph, role)]
            score = (len(mismatches), len(trim))
            if best is None or score < best[0]:
                best = (score, mapping, mismatches, trim)
        report = {'category': feature['name'], 'gt_instance': instance, 'faces': sorted(ids), 'predictions': {str(f['face_id']): f['predicted_category_name'] for f in cache['faces'] if f['face_id'] in ids}}
        if best is None:
            report['stage'] = 'insufficient_distinct_supports_or_surface_types'
        else:
            _, mapping, mismatches, trim = best
            roles = {r: set(graph.nodes[n]['face_ids']).intersection(ids) for r, n in mapping.items()}
            report.update(mapping=mapping, edge_mismatches=mismatches, trim_rejected_roles=trim,
                          cylinders=[{'faces': graph.nodes[n]['face_ids'], 'span_degrees': math.degrees(graph.nodes[n]['angular_span'])} for n in mapping.values() if graph.nodes[n]['surface_type'] == 'cylinder'])
            report['group_valid'] = core.surface_constraints_satisfied(graph, feature, mapping)
            report['active_boundary_valid'] = core.selected_patches_lie_on_active_cell(fag, graph, mapping, roles)
            report['boundary_samples'] = []
            for role, face_ids in roles.items():
                for face_id in face_ids:
                    points = core.legacy.candidate_face_sample_points(fag, face_id)
                    tolerance = 1.e-5 * max(1., max(abs(float(v)) for p in points for v in p))
                    passing = sum(all(other == role or core.oriented_support_value(graph.nodes[n], p) >= -tolerance for other, n in mapping.items()) for p in points)
                    violations = {str(other): sum(core.oriented_support_value(graph.nodes[n], p) < -tolerance for p in points) for other, n in mapping.items() if other != role}
                    report['boundary_samples'].append({'face_id': face_id, 'inside': passing, 'total': len(points), 'required': max(2, math.ceil(core.ACTIVE_PATCH_SAMPLE_FRACTION * len(points))), 'violations_by_role': violations})
            if not mismatches and not trim and report['group_valid']:
                selected, _, record = core.choose_faces(shape, fag, graph, feature, mapping)
                report['recognizer_replay'] = record
                report['selected_faces'] = sorted(selected or [])
            # Counterfactual diagnostic, not an accepted recognizer candidate.
            valid, record = core.collision_free(shape, fag, graph, feature, mapping, ids)
            report['gt_only_collision_replay'] = record
            if feature['space_model'] == 'trimmed_boundary_cell' and feature['name'] != 'circular_through_slot':
                cylinders = [graph.nodes[n] for n in mapping.values() if graph.nodes[n]['surface_type'] == 'cylinder']
                planes = [(graph.nodes[n]['geometry']['normal'], graph.nodes[n]['geometry']['offset']) for n in mapping.values() if graph.nodes[n]['surface_type'] == 'plane']
                if cylinders and planes:
                    prism = core.trimmed_patch_prism_shape(cylinders[0], fag, ids, .1)
                    clip, _ = core.legacy.build_clipped_planar_feature_space(planes, core.expanded_bounds(core.face_bounds(fag, ids), .1))
                    if prism is not None and clip is not None:
                        clipped = core.common(prism, clip)
                        report['counterfactual_plane_clipped_prism_collision_volume'] = core.shape_volume(core.common(clipped, shape)) if clipped is not None else None
            report['stage'] = ('surface_relations' if mismatches else 'surface_trim' if trim else 'group_constraint' if not report['group_valid'] else 'active_boundary' if not report['active_boundary_valid'] else 'collision' if not valid else 'passes_initial_graph_checks')
        reports.append(report)
    return reports


if __name__ == '__main__':
    original = core.extract
    context = {}
    def traced(shape, faces, fag, features):
        context.update(shape=shape, faces=faces, fag=fag, features=features)
        return original(shape, faces, fag, features)
    core.extract = traced
    core.main()
    args = core.arguments()
    output = args.output or core.OUTPUT_ROOT / args.sample
    cache = json.loads((output / 'cache.json').read_text())
    report = audit(**context, cache=cache)
    (output / 'diagnostic_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
