# vvv THOG bounded width figures in existing Instra run identities and view routes
from __future__ import annotations

from urllib.parse import parse_qs, urlparse


WIDTH_CHARTS = ('width_energy', 'width_probes', 'width_curves')


def width_figure(snapshots, chart, *, step=None, site=None):
    if chart not in WIDTH_CHARTS:
        raise ValueError('unknown width chart')
    selected = next((row for row in snapshots if row['optimizer_update'] == step), None) if step is not None else (snapshots[-1] if snapshots else None)
    steps = [row['optimizer_update'] for row in snapshots]
    if selected is None:
        return {'figure': None, 'available_steps': steps, 'available_sites': []}
    sites = selected.get('probe_residual_sites', ()) if chart == 'width_curves' else selected.get('residual_sites', ())
    sites = sites or selected.get('probe_residual_sites', ())
    sample = next((row for row in sites if row['site'] == site), None) if site else (sites[0] if sites else None)
    traces = []
    if chart == 'width_energy' and sample:
        traces.append({'type': 'scatter', 'mode': 'lines+markers', 'x': list(range(len(sample['mode_energy']))),
                       'y': sample['mode_energy'], 'name': sample['site']})
        x_label, y_label = 'Retained basis mode', 'Mean squared coefficient'
    elif chart == 'width_curves' and sample:
        for index, values in enumerate(sample.get('feature_curves', ())):
            traces.append({'type': 'scatter', 'mode': 'lines', 'x': sample['feature_indices'], 'y': values,
                           'name': f"{sample['site']} token sample {index}"})
        x_label, y_label = 'Reference feature index (bounded sample)', 'Represented feature value'
    elif chart == 'width_probes':
        probes = selected.get('probes', ())
        traces.append({'type': 'scatter', 'mode': 'lines+markers',
                       'x': [row['retained_prefix'] for row in probes], 'y': [row['delta_loss'] for row in probes],
                       'name': 'Trained-model prefix ablation'})
        if not probes:
            traces = []
        x_label, y_label = 'Retained prefix count', 'Delta validation cross-entropy (positive = harm)'
    else:
        x_label, y_label = 'Retained basis mode', 'Mean squared coefficient'
    for trace in traces:
        trace['meta'] = {'instra_workspace_optimizer_update': selected['optimizer_update'],
                         'reference_width': selected['representation']['reference_width'],
                         'residual_width': selected['representation']['residual_width'],
                         'site': None if sample is None else sample['site']}
    figure = {'data': traces, 'layout': {'xaxis': {'title': x_label}, 'yaxis': {'title': y_label},
               'margin': {'l': 65, 'r': 20, 't': 25, 'b': 60}, 'showlegend': chart == 'width_curves'}} if traces else None
    return {'figure': figure, 'available_steps': steps, 'available_sites': [row['site'] for row in sites],
            'selected_step': selected['optimizer_update'], 'selected_site': None if sample is None else sample['site'],
            'normalization_sites': selected.get('probe_normalization_sites', selected.get('normalization_sites', [])),
            'representation': selected['representation'], 'memory': selected.get('memory'),
            'baseline_validation_loss': selected.get('baseline_validation_loss'),
            'probe_interpretation': selected.get('probe_interpretation')}


def install(dashboard):
    previous_status = dashboard.RunDashboardState.status
    def status(self):
        value = previous_status(self)
        connection = self.reader._connection()
        try:
            present = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='width_snapshots'").fetchone()
            row = connection.execute('SELECT COUNT(*) AS count, MAX(optimizer_update) AS step FROM width_snapshots').fetchone() if present else None
        finally:
            connection.close()
        return {**value, 'width_snapshot_count': 0 if row is None else row['count'],
                'width_maximum_update': None if row is None else row['step']}
    dashboard.RunDashboardState.status = status
    previous_handler_for = dashboard._handler_for
    def handler_for(catalog):
        parent = previous_handler_for(catalog)
        class WidthHandler(parent):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path != '/api/width-figure':
                    return super().do_GET()
                query = parse_qs(parsed.query)
                try:
                    run_id = query.get('run', [''])[0]
                    if not run_id:
                        raise ValueError('run is required')
                    state = catalog.state_for_run(run_id)
                    step = int(query['step'][0]) if query.get('step') else None
                    chart = query.get('chart', ['width_energy'])[0]
                    with state.lock:
                        payload = width_figure(state.reader.width_snapshots(), chart, step=step, site=query.get('site', [None])[0])
                    payload['run_id'] = run_id
                    return self._send_json(payload)
                except (ValueError, TypeError) as error:
                    return self._send_json({'error': str(error)}, status=dashboard.HTTPStatus.BAD_REQUEST)
                except (FileNotFoundError, KeyError) as error:
                    return self._send_json({'error': str(error)}, status=dashboard.HTTPStatus.NOT_FOUND)
        return WidthHandler
    dashboard._handler_for = handler_for
# ^^^ THOG
