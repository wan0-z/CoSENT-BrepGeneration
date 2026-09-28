"""Single-scene Dash browser for the 100 extracted STEP feature spaces."""
from __future__ import annotations

import argparse
import html as escape_html
import logging

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from feature_viewer_geometry import ROOT, feature_boundaries, load_sample, normal_arrow, sample_index, topology_meshes

COLORS = ['#137eab','#e27a24','#239b78','#a65da5','#d65565','#5474c5','#a59120','#3b9caa','#ad6743','#775cbd']


def color(instance_id):
    return COLORS[int(instance_id) % len(COLORS)]


def mesh_trace(mesh, shade, opacity, name, hover, meta):
    if not mesh['triangles']:
        return None
    x,y,z = zip(*mesh['vertices']); i,j,k = zip(*mesh['triangles'])
    return go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,color=shade,opacity=opacity,
        flatshading=True,name=name,showlegend=False,meta=meta,
        hovertemplate=hover+'<extra></extra>',
        lighting=dict(ambient=.68,diffuse=.72,specular=.12),
        lightposition=dict(x=120,y=180,z=160))


def feature_options(sample):
    data = load_sample(sample)
    options = [dict(label='全部特征',value='all')]
    options.extend(dict(label=f"#{f['id']}  {f['name']} · {len(set(i for r in f['roles'] for i in r['source_face_ids']))} faces",
                        value=str(f['id'])) for f in data['feature_spaces'])
    first = str(data['feature_spaces'][0]['id']) if data['feature_spaces'] else 'all'
    return options, first


def make_figure(sample, selection, layers, extension=.3, opacity=.24):
    data = load_sample(sample); geometry = topology_meshes(sample)
    selected = [f for f in data['feature_spaces'] if selection == 'all' or str(f['id']) == str(selection)]
    byface = {}
    for f in selected:
        for fid in {fid for r in f['roles'] for fid in r['source_face_ids']}:
            byface.setdefault(fid,[]).append(f)
    figure = go.Figure(); bounds = list(data['bbox']); empty_roles = []; infeasible = []; approximation = False
    if 'faces' in layers or 'context' in layers:
        for face in geometry['faces']:
            matched = byface.get(face['face_id'],[])
            if matched and 'faces' not in layers: continue
            if not matched and 'context' not in layers: continue
            label = ' / '.join(f"#{f['id']} {f['name']}" for f in matched) or '背景模型'
            trace = mesh_trace(face,color(matched[0]['id']) if matched else '#acb8c5',
                .96 if matched else .10, label, f"Face {face['face_id']}<br>{label}",
                dict(kind='topology',face_id=face['face_id'],features=[f['id'] for f in matched]))
            if trace is not None: figure.add_trace(trace)
        # True trimmed STEP edge curves, not triangulation wireframes.
        line_groups = {}
        for edge in geometry['edges']:
            matched = [f for fid in edge['face_ids'] for f in byface.get(fid,[])]
            if matched and 'faces' not in layers: continue
            if not matched and 'context' not in layers: continue
            key = ('selected',matched[0]['id']) if matched else ('context',-1)
            line_groups.setdefault(key,[]).extend(edge['points']+[[None,None,None]])
        for (kind,fid),points in line_groups.items():
            x,y,z = zip(*points)
            figure.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',showlegend=False,
                line=dict(color=color(fid) if kind=='selected' else '#94a1af',width=3 if kind=='selected' else 1),
                opacity=.95 if kind=='selected' else .22,hoverinfo='skip',meta=dict(kind='edge')))
    if 'spaces' in layers or 'normals' in layers:
        for f in selected:
            boundary = feature_boundaries(sample,f['id'],float(extension))
            if not boundary['planar_feasible']:
                infeasible.append(str(f['id']))
                continue
            approximation |= boundary['approximate']
            for a in range(3):
                bounds[a] = min(bounds[a],boundary['bounds'][a])
                bounds[a+3] = max(bounds[a+3],boundary['bounds'][a+3])
            for patch in boundary['patches']:
                role,mesh = patch['role'],patch['mesh']
                if not mesh['triangles']:
                    empty_roles.append(f"#{f['id']}/{role['role']}"); continue
                label = f"#{f['id']} {f['name']} · {role['role']}"
                hover = f"{label}<br>特征空间边界 · {role['parameters']['type']}<br>sigma = {role['sigma']}"
                if 'spaces' in layers:
                    trace = mesh_trace(mesh,color(f['id']),opacity,label,hover,
                        dict(kind='feature_space',feature_id=f['id'],role_id=role['role_id']))
                    if trace is not None: figure.add_trace(trace)
                if 'normals' in layers:
                    anchor,direction = normal_arrow(role,mesh)
                    length = boundary['span']*.10; tip = anchor+direction*length
                    figure.add_trace(go.Scatter3d(x=[anchor[0],tip[0]],y=[anchor[1],tip[1]],z=[anchor[2],tip[2]],
                        mode='lines',line=dict(color=color(f['id']),width=5),showlegend=False,hoverinfo='skip'))
                    figure.add_trace(go.Cone(x=[tip[0]],y=[tip[1]],z=[tip[2]],u=[direction[0]],v=[direction[1]],w=[direction[2]],
                        sizemode='absolute',sizeref=length*.28,anchor='tip',showscale=False,
                        colorscale=[[0,color(f['id'])],[1,color(f['id'])]],
                        hovertemplate=f"{role['role']} · 法向指向 void side<extra></extra>",
                        meta=dict(kind='normal',feature_id=f['id'],role_id=role['role_id'])))
    # One proportional scene; keep camera during layer/feature adjustments.
    span = np.array(bounds[3:])-bounds[:3]; pad = max(float(max(span))*.025,1e-6)
    axes = {f'{a}axis':dict(visible=False,range=[bounds[i]-pad,bounds[i+3]+pad],autorange=False) for i,a in enumerate('xyz')}
    figure.update_layout(scene=dict(**axes,aspectmode='data',bgcolor='#f7f9fc',dragmode='orbit',
                         camera=dict(eye=dict(x=.95,y=1.1,z=.8))),
        margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='#f7f9fc',hovermode='closest',
        uirevision=f'feature-viewer-{sample}',font=dict(family='Arial, sans-serif',color='#23364a'))
    status = f"{sample}.step · {len(data['feature_spaces'])} 个提取实例 · 当前 {len(selected)} 个实例 / {len(byface)} 个特征面"
    note = '实色为原始拓扑面，半透明为特征空间边界；箭头指向 void side。显示范围仅用于裁切，不添加封口。'
    if approximation and ('spaces' in layers or 'normals' in layers): note += ' 曲面交线为网格近似。'
    if infeasible: note += ' 实例 #'+', #'.join(infeasible)+' 的平面法向约束互斥，无法形成共同 void-side 空间；保留原特征面，不伪造空间。'
    if empty_roles: note += ' 当前约束下无可见边界的角色：'+', '.join(empty_roles[:8])
    return figure,status,note


def make_app():
    app = Dash(__name__,title='Feature space · Transformer demo',assets_folder=str(ROOT/'viewer_assets'))
    names = list(sample_index()); default = names[0]
    options,first = feature_options(default)
    app.layout = html.Main([
        html.Header([html.Div([html.Div('TRANSFORMER DEMO',className='eyebrow'),html.H1('特征面与特征空间')]),
                     html.Span(f'{len(names)} STEP samples',className='dataset-tag')]),
        html.Div([
            html.Div([html.Label('数据样本',htmlFor='sample'),dcc.Dropdown(id='sample',options=[dict(label=n+'.step',value=n) for n in names],value=default,clearable=False)],className='sample-control'),
            html.Div([html.Label('特征实例',htmlFor='feature'),dcc.Dropdown(id='feature',options=options,value=first,clearable=False)],className='feature-control'),
            html.Div([html.Label('显示图层'),dcc.Checklist(id='layers',options=[dict(label='特征面',value='faces'),dict(label='特征空间',value='spaces'),dict(label='法向',value='normals'),dict(label='背景模型',value='context')],
                value=['faces','spaces','normals','context'],inline=True)],className='layer-control'),
        ],className='toolbar'),
        html.Div([html.Div([html.Label('空间延展'),dcc.Slider(id='extension',min=.05,max=1.,step=.05,value=.3,marks={.05:'近',.3:'0.3×',1.:'远'},tooltip={'placement':'bottom','always_visible':False})]),
                  html.Div([html.Label('空间透明度'),dcc.Slider(id='opacity',min=.05,max=.65,step=.05,value=.25,marks={.05:'淡',.25:'0.25',.65:'浓'})]),
                  html.Div(id='status',className='status',role='status')],className='view-controls'),
        dcc.Loading(dcc.Graph(id='scene',figure=go.Figure(),className='scene',config=dict(scrollZoom=True,displaylogo=False,
             modeBarButtonsToRemove=['lasso2d','select2d'],toImageButtonOptions=dict(format='png',filename='feature-space'))),type='circle',color='#137eab'),
        html.Footer(id='note'),
    ])

    @app.callback(Output('feature','options'),Output('feature','value'),Input('sample','value'))
    def change_sample(sample):
        return feature_options(sample)

    @app.callback(Output('scene','figure'),Output('status','children'),Output('note','children'),
                  Input('sample','value'),Input('feature','value'),Input('layers','value'),Input('extension','value'),Input('opacity','value'))
    def display(sample,selection,layers,extension,opacity):
        try:
            return make_figure(sample,selection,layers,float(extension),float(opacity))
        except Exception as exc:
            logging.exception('Feature viewer failed for %s',sample)
            figure = go.Figure()
            figure.add_annotation(text='无法加载此样本',showarrow=False)
            return figure,'加载失败',escape_html.escape(str(exc))
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8057)
    args = parser.parse_args()
    make_app().run(host='127.0.0.1',port=args.port,debug=False)


if __name__ == '__main__': main()
