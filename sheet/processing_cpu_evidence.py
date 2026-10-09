# vvv THOG CPU work and CUDA copies are independent evidence, never GPU kernels or split device metrics
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

IDENTITIES = ('snapshot_id', 'cpu_task_id', 'cpu_matrix_id', 'upload_id', 'matrix_use_id', 'source_name', 'source_offset')


def union_duration(intervals):
    end, total = None, 0.0
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop-start
            end = stop
        elif stop > end:
            total += stop-end
            end = stop
    return total


def copy_activities(connection, tables, correlations, capture_start, capture_end):
    rows, supported = [], False
    for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMCPY2'):
        if table not in tables:
            continue
        names = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
        if not {'start', 'end', 'copyKind'}.issubset(names):
            continue
        supported = True
        cursor = connection.execute(f'SELECT * FROM "{table}"')
        columns = [item[0] for item in cursor.description]
        for index, values in enumerate(cursor):
            activity = dict(zip(columns, values))
            start, end = activity['start'], activity['end']
            if start is None or end is None or end <= capture_start or start >= capture_end:
                continue
            direction = {1:'H2D', 2:'D2H'}.get(activity['copyKind'], 'OTHER')
            op = correlations.get(activity.get('correlationId'), {})
            clipped_start, clipped_end = max(start,capture_start), min(end,capture_end)
            duration_us = (clipped_end-clipped_start)/1000
            size = activity.get('bytes', activity.get('size'))
            censored = start < capture_start or end > capture_end
            rows.append({
                'activity_id':f'{table}:{index}', 'direction':direction, 'raw_start_ns':start, 'raw_end_ns':end,
                'start_us':(clipped_start-capture_start)/1000, 'end_us':(clipped_end-capture_start)/1000,
                'duration_us':duration_us, 'left_censored':start<capture_start, 'right_censored':end>capture_end,
                'bytes':size, 'full_transfer_bandwidth_gb_s':None if censored or not size or duration_us<=0 else size/duration_us/1000,
                'stream':activity.get('streamId'), 'context_id':activity.get('contextId'), 'device_id':activity.get('deviceId'),
                'correlation_id':activity.get('correlationId'), 'logical_identity_known':bool(op.get('snapshot_id') or op.get('upload_id')),
                **{key:op.get(key) for key in (*IDENTITIES,'family','layer','operation')},
            })
    return rows, supported


def host_clock(capture_metadata, gpu_start):
    lower, upper = capture_metadata.get('host_nvtx_lower_ns'), capture_metadata.get('host_nvtx_upper_ns')
    if lower is None or upper is None or upper < lower or (upper-lower)/2e6 > 5:
        return None, None
    return (lower+upper)//2, (upper-lower)/2e6


def cpu_task_rows(events, capture_metadata, gpu_start, gpu_end):
    origin, uncertainty = host_clock(capture_metadata, gpu_start)
    tasks = {}
    for event in events:
        identity = event.get('cpu_matrix_id')
        if not identity or event.get('event') not in ('cpu_task_start','cpu_matrix_ready'):
            continue
        tasks.setdefault(identity, {}).update(event)
    duration_ns = gpu_end-gpu_start
    rows = []
    for identity,event in tasks.items():
        start, end = event.get('start_ns'), event.get('end_ns')
        known = origin is not None and start is not None and end is not None
        start_us = max(start-origin,0)/1000 if known else None
        end_us = min(end-origin,duration_ns)/1000 if known else None
        before = known and end <= origin
        after = known and start >= origin+duration_ns
        rows.append({**{key:value for key,value in event.items() if key not in ('event','tensor')},
            'cpu_matrix_id':identity, 'clock_alignment_known':known, 'clock_uncertainty_ms':uncertainty,
            'start_us':None if before or after else start_us, 'end_us':None if before or after else end_us,
            'duration_us':None if not known or before or after else max(0,end_us-start_us),
            'cpu_service_us':None if start is None or end is None else (end-start)/1000,
            'precursor':bool(before), 'outside_capture':bool(before or after),
            'left_censored':bool(known and start < origin < end),
            'right_censored':bool(known and start < origin+duration_ns < end),
        })
    return rows


def export_rows(path, rows, minimum_fields=()):
    fields = list(dict.fromkeys([*minimum_fields,*(key for row in rows for key in row)]))
    with path.open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list,tuple)) else value for key,value in row.items()})


def extend_processing(data, output_directory, prefix, report, copies, copies_supported, capture_start, capture_end):
    metadata=data['metadata']
    capture=metadata.get('capture',{})
    events=list((report or {}).get('cpu_lifecycle',()))
    update=capture.get('optimizer_update')
    uses=[event for event in events if event.get('event')=='matrix_use' and (update is None or event.get('optimizer_step')==update)]
    snapshots={event.get('snapshot_id') for event in uses}
    relevant=[event for event in events if (update is None or event.get('optimizer_step')==update or event.get('snapshot_id') in snapshots)]
    tasks=cpu_task_rows(relevant,capture,capture_start,capture_end)
    memory=[event for event in relevant if event.get('event') in ('cpu_memory','gpu_storage_released','upload_queued','upload_submitted','upload_complete_observed','late_copy')]
    targeted=[event for event in uses if event.get('targeted',True)]
    outcomes=Counter(event.get('final_outcome','unknown') for event in targeted)
    fallback=Counter(event.get('fallback_reason','unknown') for event in targeted if event.get('final_outcome')=='COMPLETE MISS')
    reused=Counter(event.get('cpu_matrix_id') for event in uses if event.get('cpu_matrix_id'))
    main=[(row['start_us'],row['end_us']) for row in data.get('intervals',()) if row.get('owner')=='MAIN']
    for copy in copies:
        overlap=[(max(copy['start_us'],a),min(copy['end_us'],b)) for a,b in main if min(copy['end_us'],b)>max(copy['start_us'],a)]
        copy['main_overlap_us']=union_duration(overlap)
        copy['overlaps_main']=bool(overlap)
    for row in data.get('stream_resources',()):
        row['copy_activity_present']=any(copy['end_us']>float(row.get('start_us',0)) and copy['start_us']<float(row.get('end_us',0)) for copy in copies)
    runtime=(report or {}).get('cpu_runtime',{})
    data.update(cpu_tasks=tasks, transfers=copies, cpu_lifecycle=relevant, cpu_memory=memory, cpu_runtime=runtime,
                cpu_configuration=(report or {}).get('cpu_configuration',metadata.get('run',{}).get('config',{})))
    data['cpu_summary']={
        'evidence_available':report is not None, 'cuda_copy_coverage':'available' if copies_supported else 'unknown',
        'eligible_matrix_uses':len(targeted) if report is not None else None,
        'not_targeted_uses':len(uses)-len(targeted) if report is not None else None,
        'full_hits':outcomes['FULL HIT'] if report is not None else None,
        'complete_misses':outcomes['COMPLETE MISS'] if report is not None else None,
        'fallback_reasons':dict(fallback), 'unique_cpu_matrices':len(tasks) if report is not None else None,
        'cpu_service_ms':sum(row['cpu_service_us'] or 0 for row in tasks)/1000 if report is not None else None,
        'cpu_matrix_use_counts':dict(reused), 'phase_counts':dict(Counter(event.get('phase','unknown') for event in uses)),
        'h2d_busy_us':union_duration((row['start_us'],row['end_us']) for row in copies if row['direction']=='H2D') if copies_supported else None,
        'd2h_busy_us':union_duration((row['start_us'],row['end_us']) for row in copies if row['direction']=='D2H') if copies_supported else None,
        'late_upload_count':len({event.get('upload_id') for event in relevant if event.get('event')=='late_copy'}),
        'gpu_co_residency':'N/A: CPU materialisation', 'accounting':'CPU and COPY are overlays; device metrics and exclusive update phase totals are unchanged',
    }
    metadata.update(schema_version=5, materialisation_device='cpu_and_gpu', gpu_co_residency='N/A: CPU materialisation', copy_coverage=data['cpu_summary']['cuda_copy_coverage'])
    if report is None:
        metadata['warnings'].append('CPU lifecycle evidence is missing; CPU work, matrix-use and hit totals are unknown')
    datasets={'transfers':copies,'cpu_tasks':tasks,'cpu_memory':memory,'cpu_lifecycle':relevant}
    for key,rows in datasets.items():
        filename=f'{prefix}processing_{key}.csv'
        metadata['files'][key]=filename
        export_rows(Path(output_directory)/filename,rows,('activity_id',) if key=='transfers' else ('event',))
    data['premat_compatibility']={'available':False,'reason':'N/A: CPU materialisation','rows':[]}
    return data


def update_timing_overlay(runtime, optimizer_update, host_origin_ns):
    if runtime is None:
        return {'cpu_overlay':{'available':False,'reason':'CPU runtime unavailable','additive':False}}
    report=runtime.report()
    events=[event for event in report.get('cpu_lifecycle',()) if event.get('optimizer_step')==optimizer_update]
    tasks={}
    for event in events:
        if event.get('cpu_matrix_id') and event.get('event')=='cpu_matrix_ready':
            tasks[event['cpu_matrix_id']]=event
    rows=[]
    for event in tasks.values():
        rows.append({**event,'host_start_ms':None if host_origin_ns is None else (event['start_ns']-host_origin_ns)/1e6,
                     'host_end_ms':None if host_origin_ns is None else (event['end_ns']-host_origin_ns)/1e6})
    return {'cpu_overlay':{'available':True,'additive':False,'tasks':rows,'lifecycle':events,
            'cpu_runtime':report.get('cpu_runtime',{}),'description':'CPU tasks and transfer/readiness boundaries overlay the exclusive host phases; they are not additional phase time'}}
# ^^^ THOG
