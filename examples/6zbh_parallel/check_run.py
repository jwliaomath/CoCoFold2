"""CPU output audit for a fresh complete-epoch parallel run; no model decoding."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone, timedelta


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def local_path(root, relative):
    path = (root/relative).resolve()
    path.relative_to(root.resolve())
    return path


def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def read_rows(path, event=None, limit=None):
    """Read only the audited prefix; a later concurrently written tail is irrelevant."""
    rows=[]
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row=json.loads(line)
            if event is None or row.get('event')==event:
                rows.append(row)
                if limit is not None and len(rows)==limit:
                    break
    return rows


def check_run(output, epochs, completed_prefix=False, reason=None):
    out = Path(output).resolve()
    checks, performance = [], {}
    acceptance=dict(mode='completed_epoch_prefix' if completed_prefix else 'full_run',
        requested_epochs=epochs,original_target_epochs=None,reason=reason,
        checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        observed_rank_records=[],observed_launcher=None,
        job_state_note='Recorded statuses are snapshots, not scheduler/live-process checks; running can remain stale after a kill.',
        whole_job_completion_claimed=False)
    def check(name, condition, **detail):
        checks.append(dict(name=name,passed=bool(condition),**detail))
    try:
        if completed_prefix and (not isinstance(reason,str) or not reason.strip()):
            raise ValueError('Prefix acceptance requires an explicit reason')
        import torch
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from coordinate_transform import template_atom_keys, CoordinateTransform
        from checkpoint_sampling import global_coordinates
        from structure_io import read_template
        final = read_json(out/f'model_epoch_{epochs}.json')
        if final.get('format') != 'cocofold2-parallel-epoch-v1':
            raise ValueError('Unsupported parallel epoch index')
        groups = final['group_identity']
        ranks = [g['rank'] for g in groups]
        if ranks != list(range(len(groups))) or len(groups) != 2:
            raise ValueError('This example expects two ranks in rank order')
        histories, timings = [], []
        particle_count = None
        batch_size = final['science_args']['batch_size']
        if not isinstance(batch_size,int) or batch_size <= 0:
            raise ValueError('Invalid saved batch size')
        for rank in ranks:
            record = out/'records'/f'rank{rank}'
            summary = read_json(record/'run_summary.json')
            acceptance['observed_rank_records'].append(dict(rank=rank,status=summary.get('status'),
                error=summary.get('error'),ended_at=summary.get('ended_at'),elapsed_seconds=summary.get('elapsed_seconds')))
            if not completed_prefix:
                check(f'rank{rank} finite run duration',finite_number(summary.get('elapsed_seconds')) and summary['elapsed_seconds']>=0)
            resolved = read_json(record/'resolved_config.json')
            stages = {r['stage']:r['values'] for r in resolved['stages']}
            reports = stages['parallel_inputs']
            counts = {r['n_particles'] for r in reports}
            if len(counts) != 1 or next(iter(counts)) <= 0:
                raise ValueError('Invalid or inconsistent particle counts')
            count = next(iter(counts))
            if particle_count is not None and particle_count != count:
                raise ValueError('Particle count differs between rank records')
            particle_count = count
            if not completed_prefix:
                check(f'rank{rank} run completed',summary['status']=='success' and summary.get('error') is None)
            requested = read_json(record/'requested_config.json')['arguments']
            if acceptance['original_target_epochs'] is None:
                acceptance['original_target_epochs']=requested['epochs']
            check(f'rank{rank} original target consistent',requested['epochs']==acceptance['original_target_epochs'])
            check(f'rank{rank} complete example configuration',(requested['epochs']>=epochs if completed_prefix else requested['epochs']==epochs) and not requested.get('resume')
                  and requested.get('max_steps') is None and stages['parallel_science']==final['science_args'])
            command = read_json(record/'command.json')
            check(f'rank{rank} command and provenance',bool(command.get('argv')) and
                  bool(read_json(record/'provenance.json')['source_sha256']) and
                  bool(read_json(record/'run_identity.json')['run_id']))
            if requested.get('submission_script'):
                check(f'rank{rank} submission script', (record/'submitted_script.sh').is_file() and
                      read_json(record/'submission_script.json')['sha256']==digest(record/'submitted_script.sh'))
            total_steps = epochs*math.ceil(count/batch_size)
            steps=read_rows(record/('metrics.jsonl' if rank==0 else f'metrics_rank{rank}.jsonl'),
                            event='train_step',limit=total_steps if completed_prefix else None)
            check(f'rank{rank} recorded progress',[r['global_step'] for r in steps]==list(range(1,total_steps+1)))
            check(f'rank{rank} finite training metrics',bool(steps) and all(
                finite_number(r['total_loss']) and finite_number(r['elapsed_seconds']) and r['elapsed_seconds']>=0 for r in steps))
            histories.append([r['particle_indices'] for r in steps])
            for epoch in range(epochs):
                epoch_steps = [r for r in steps if r['epoch']==epoch]
                consumed = [i for r in epoch_steps for i in r['particle_indices']]
                check(f'rank{rank} epoch{epoch+1} full STAR coverage', sorted(consumed)==list(range(count)) and
                      [r['batch'] for r in epoch_steps]==list(range(math.ceil(count/batch_size))))
            detailed=read_rows(out/f'chain_parallel_2d_metrics_rank{rank}.jsonl',limit=total_steps if completed_prefix else None)
            check(f'rank{rank} performance rows',len(detailed)==total_steps and
                [(r['epoch'],r['batch']) for r in detailed]==[(r['epoch'],r['batch']) for r in steps])
            durations = [r['batch_time_seconds'] for r in detailed]
            check(f'rank{rank} finite performance',bool(durations) and all(finite_number(v) and v>=0 for v in durations))
            env = read_json(record/'environment.json')
            gpu = str(env.get('process',{}).get('WORLD_SIZE',''))=='2' and requested.get('backend')=='nccl'
            for name in ('peak_allocated_memory_mb','peak_reserved_memory_mb'):
                values = [r.get(name) for r in detailed]
                check(f'rank{rank} {name}', all(finite_number(v) and v>=0 for v in values) if gpu
                      else all(v is None or (finite_number(v) and v>=0) for v in values))
            performance[f'rank{rank}'] = dict(
                devices=env.get('devices'), run_elapsed_seconds=summary['elapsed_seconds'] if not completed_prefix and finite_number(summary.get('elapsed_seconds')) else None,
                run_duration_note='Whole-job time is not attributed to the accepted prefix.' if completed_prefix else 'Recorded whole rank invocation duration.',
                step_mean_seconds=statistics.mean(durations) if durations and all(finite_number(v) for v in durations) else None,
                step_max_seconds=max((v for v in durations if finite_number(v)),default=None),
                peak_allocated_memory_MiB=max((r['peak_allocated_memory_mb'] for r in detailed if finite_number(r.get('peak_allocated_memory_mb'))),default=None),
                peak_reserved_memory_MiB=max((r['peak_reserved_memory_mb'] for r in detailed if finite_number(r.get('peak_reserved_memory_mb'))),default=None))
            timings.append(durations)
        check('rank particle orders match',histories[0]==histories[1])
        steps_per_epoch = math.ceil(particle_count/batch_size)
        atom_keys = [key for group in groups for key in group['global_atom_keys']]
        for epoch in range(1,epochs+1):
            index = read_json(out/f'model_epoch_{epoch}.json')
            check(f'epoch{epoch} index complete',index['format']==final['format'] and index['epoch_complete'] and
                  index['next_epoch']==epoch and index['global_step']==epoch*steps_per_epoch and
                  index['group_identity']==groups and index['science_args']==final['science_args'] and
                  [(r['rank'],r['id']) for r in index['components']]==[(g['rank'],g['id']) for g in groups])
            merged = local_path(out,index['merged_cif'])
            check(f'epoch{epoch} merged identity',template_atom_keys(merged)==atom_keys)
            check(f'epoch{epoch} finite merged coordinates',torch.isfinite(read_template(merged)[0]).all())
            for row,group in zip(index['components'],groups):
                path, cif = local_path(out,row['path']),local_path(out,row['cif'])
                check(f'epoch{epoch} rank{row["rank"]} files',path.is_file() and path.stat().st_size>0 and cif.is_file())
                keys = template_atom_keys(cif)
                mapped = [[group['chain_id_map'][k[0]],*k[1:]] for k in keys]
                check(f'epoch{epoch} rank{row["rank"]} structure',mapped==group['global_atom_keys'] and
                      torch.isfinite(read_template(cif)[0]).all())
        for row in final['components']:
            rank = row['rank']
            path = local_path(out,row['path'])
            check(f'rank{rank} final checkpoint hash',digest(path)==row['sha256'])
            cache = torch.load(path,map_location='cpu',weights_only=False)
            progress = cache['training_progress']
            check(f'rank{rank} final checkpoint progress',progress['epoch_complete'] and progress['epoch']==epochs and
                  progress['global_step']==epochs*steps_per_epoch and cache['parallel_resume']['epoch_complete'])
            check(f'rank{rank} finite latent',torch.isfinite(cache['z_bias']).all())
            check(f'rank{rank} finite GMM',all(torch.isfinite(v).all() for v in cache['gmm']['state_dict'].values()))
            states = cache['opt_state']['state'].values()
            check(f'rank{rank} optimizer progress',bool(cache['opt_state']['state']) and all(int(s['step'])==epochs*steps_per_epoch
                and torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() for s in states))
            raw = cache['pred_dict']['coordinate'][0]
            if cache.get('coordinate_transform') is not None:
                transform=CoordinateTransform.from_checkpoint(cache['coordinate_transform'],local_path(out,row['cif']),'cpu')
                expected=transform(raw).float()
            else:
                expected = global_coordinates(cache,raw).float()
            actual = read_template(local_path(out,row['cif']))[0]
            mse = float((actual-expected).square().mean())
            check(f'rank{rank} final saved coordinate consistency',math.isfinite(mse) and mse<1e-8,mse_A2=mse if math.isfinite(mse) else None)
            del cache
        maxima = [max(values) for values in zip(*timings) if all(finite_number(v) for v in values)]
        performance['distributed_steps'] = dict(count=len(maxima),
            sum_max_rank_seconds=sum(maxima),mean_max_rank_seconds=statistics.mean(maxima) if maxima else None,
            definition='Per step, maximum rank duration; excludes model initialization, epoch export and file I/O. Not whole-job wall time.')
        if (out/'launcher.json').is_file():
            launcher = read_json(out/'launcher.json')
            acceptance['observed_launcher']={k:launcher.get(k) for k in ('status','error','epochs','ended_at','elapsed_seconds')}
            if not completed_prefix:
                check('launcher completed',launcher['status']=='success')
                performance['launcher_elapsed_seconds'] = launcher.get('elapsed_seconds')
        error = None
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    passed = bool(checks) and all(c['passed'] for c in checks) and error is None
    acceptance['whole_job_completion_claimed']=passed and not completed_prefix
    acceptance['statement']=(f"Author revised acceptance from original target {acceptance['original_target_epochs']} epochs "
        f"to the first {epochs} complete epochs. Later work and whole-job success are outside this audit."
        if completed_prefix else 'Full original run completion audit.')
    return dict(schema_version=1,passed=passed,total=len(checks),passed_count=sum(c['passed'] for c in checks),
        failed_count=sum(not c['passed'] for c in checks),error=error,checks=checks,performance=performance,
        acceptance=acceptance,
        scope='First requested complete epochs only; not whole-job completion' if completed_prefix else 'Completed fresh parallel example; file/record audit, no model execution',
        checkpoint_hash_scope='Final epoch only; all epochs checked for index, file existence and CIF identities/finite coordinates',
        structure_quality='pending author review: map agreement and RMSD are not automated pass criteria')


def write_report(output, report):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    # Prefix audits never overwrite the running launcher's full-run summary.
    acceptance=report.get('acceptance',{})
    stem=f"acceptance_epoch_{acceptance['requested_epochs']}" if acceptance.get('mode')=='completed_epoch_prefix' else 'summary'
    (output/(stem+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = [f"Automatic checks passed: {report['passed']}; {report['passed_count']}/{report['total']}",
             *[f"{'PASS' if c['passed'] else 'FAIL'}: {c['name']}" for c in report['checks']],
             f"Error: {report['error']}",acceptance.get('statement',''),f"Reason: {acceptance.get('reason')}",report['structure_quality']]
    (output/(stem+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('passed','total','passed_count','failed_count','error','performance','acceptance','structure_quality')},indent=2))
