# vvv THOG verify durable JSON deletion receipts, recovery, host backlog and acquisition ordering
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import instra_monitoring
import instra_network
from tests import test_instra_monitoring_unittest as monitoring_fixture
FakeNetwork = monitoring_fixture.FakeNetwork
dashboard = monitoring_fixture.dashboard
from sheet.local_chart_store import LocalChartStore


class DistributedDeletionTests(unittest.TestCase):
    setUp = monitoring_fixture.MonitoringTests.setUp
    _restore = monitoring_fixture.MonitoringTests._restore
    _producer = monitoring_fixture.MonitoringTests._producer

    def producer_service(self, host_id, path, offline=False):
        network = FakeNetwork(self.root)
        network.local_id = host_id
        network.add_host(self.network.local_id, self.root / 'viewer')
        if offline:
            network.add_host('thog_host.offline', self.root / 'offline')
        catalog = object.__new__(dashboard.DashboardCatalog)
        self.original['__init__'](catalog, root=path.parent.parent.parent)
        service = instra_monitoring.MonitoringService(network, catalog,
            storage_root=self.root / ('producer_' + host_id), start_worker=False)
        catalog.monitoring = service
        self.addCleanup(service.close)
        return service, catalog

    def setup_deletion(self, *, acquired=True, offline=False):
        host_id, path = self._producer('producer')
        if acquired:
            self.monitor._sync_host(host_id)
        service, catalog = self.producer_service(host_id, path, offline)
        run_id = catalog.runs()['runs'][0]['dashboard_run_id']
        notice = service.begin_authoritative_delete(run_id)
        self.uploads = []
        self.fail_upload = False
        def upload(host, relative, payload):
            if self.fail_upload:
                raise instra_network.NetworkError('transfer', 'receipt upload failed', host)
            target = Path(self.network._host(host)['last_discovered']['instra_logs_root']) / relative
            instra_monitoring._atomic_json(target, payload)
            self.uploads.append(payload)
        self.network.monitor_upload_receipt = upload
        return host_id, path, service, catalog, notice

    def wait_idle(self, host_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with self.monitor.lock:
                if host_id not in self.monitor.active:
                    return
            time.sleep(.01)
        self.fail('background cleanup remained active')

    def test_single_run_hidden_then_confirmed_without_touching_original_logs(self):
        host, path, producer, catalog, request = self.setup_deletion()
        self.assertEqual(catalog.runs()['runs'], [])
        self.assertTrue(path.exists())
        original_log = path.parent.parent / 'train.log'
        self.monitor._sync_host(host)
        self.assertEqual(self.catalog.runs()['runs'], [])
        self.assertEqual(len(self.uploads), 1)
        self.assertEqual(producer._finalize_deletions(), [request['deletion_request_id']])
        self.assertFalse(path.exists())
        self.assertTrue(original_log.exists())
        self.assertTrue((self.root / 'producer/wandb/run-same_id/run-same_id.wandb').exists())
        self.assertFalse(list((catalog.root / '.instra_deletions/notices').glob('*.json')))

    def test_no_copy_receipt_is_identical_and_prevents_first_acquisition(self):
        host, path, producer, _catalog, _request = self.setup_deletion(acquired=False)
        self.monitor._sync_host(host)
        self.assertTrue(self.uploads[0]['deletion_processed'])
        self.assertNotIn('copy_existed', self.uploads[0])
        self.assertFalse(any(call[-1] for call in self.network.transfer_calls))
        self.assertEqual(self.catalog.runs()['runs'], [])
        producer._finalize_deletions()
        self.assertFalse(path.exists())

    def test_partial_receipts_survive_restart_and_offline_host_backlog(self):
        host, path, producer, catalog, request = self.setup_deletion(offline=True)
        self.monitor._sync_host(host)
        self.assertEqual(producer._finalize_deletions(), [])
        self.assertEqual(producer.deletion_snapshot()['pending'][0]['outstanding_hosts'], ['offline'])
        replacement = instra_monitoring.MonitoringService(producer.network, catalog,
            storage_root=producer.storage_root, start_worker=False)
        self.addCleanup(replacement.close)
        catalog.monitoring = replacement
        self.assertTrue(path.exists())
        self.assertEqual(replacement.deletion_snapshot()['pending'][0]['requested_at'],
                         producer.deletion_snapshot()['pending'][0]['requested_at'])
        offline = FakeNetwork(self.root)
        offline.local_id = 'thog_host.offline'
        offline.add_host(host, self.root / 'producer')
        offline.monitor_upload_receipt = self.network.monitor_upload_receipt
        offline_catalog = object.__new__(dashboard.DashboardCatalog)
        self.original['__init__'](offline_catalog, root=self.root / 'offline_local')
        viewer = instra_monitoring.MonitoringService(offline, offline_catalog,
            storage_root=self.root / 'offline_storage', start_worker=False)
        self.addCleanup(viewer.close)
        offline_catalog.monitoring = viewer
        viewer._sync_host(host)
        self.assertEqual(replacement._finalize_deletions(), [request['deletion_request_id']])

    def test_timeout_is_captured_and_expiry_is_logged_separately(self):
        _host, path, producer, _catalog, request = self.setup_deletion(offline=True)
        producer.set_deletion_timeout_days(30)
        item = producer.deletions[request['deletion_request_id']]
        self.assertEqual(item['timeout_days'], 7)
        requested_epoch = instra_monitoring.datetime.fromisoformat(item['requested_at']).timestamp()
        with patch.object(instra_monitoring.time, 'time', return_value=requested_epoch + 7 * 86400 - 1):
            self.assertEqual(producer._finalize_deletions(), [])
        with patch.object(instra_monitoring.time, 'time', return_value=requested_epoch + 7 * 86400 + 1):
            self.assertEqual(producer._finalize_deletions(), [request['deletion_request_id']])
        self.assertFalse(path.exists())
        self.assertEqual(self.network.events[-1][0][3], 'timeout')

    def test_repeated_request_does_not_reset_original_clock_or_expected_hosts(self):
        _host, _path, producer, catalog, request = self.setup_deletion()
        item = producer.deletions[request['deletion_request_id']]
        producer.set_deletion_timeout_days(20)
        again = producer.begin_authoritative_delete(item['run_id'])
        self.assertEqual(again['deletion_request_id'], request['deletion_request_id'])
        self.assertEqual(len(producer.deletions), 1)
        self.assertEqual(item['timeout_days'], 7)
        self.assertEqual(catalog.runs()['runs'], [])

    def test_missing_notice_is_republished_after_restart(self):
        _host, _path, producer, catalog, request = self.setup_deletion()
        notice = catalog.root / '.instra_deletions/notices' / (request['deletion_request_id'] + '.json')
        notice.unlink()
        producer._finalize_deletions()
        self.assertTrue(notice.exists())

    def test_mismatched_malformed_and_unsuccessful_receipts_do_not_confirm(self):
        host, path, producer, catalog, request = self.setup_deletion()
        self.monitor._sync_host(host)
        target = catalog.root / '.instra_deletions/receipts' / request['deletion_request_id'] / 'thog_host_viewer.json'
        correct = json.loads(target.read_text())
        for key, bad in [('producer_host_id', 'wrong'), ('run_id', 'wrong'), ('deletion_request_id', 'wrong'),
                         ('responding_host_id', 'thog_host.unlisted'), ('source_chart_path', 'wrong/charts.sqlite3'),
                         ('deletion_processed', False), ('schema', 99)]:
            with self.subTest(key=key):
                target.write_text(json.dumps({**correct, key: bad}))
                self.assertEqual(producer._finalize_deletions(), [])
                self.assertTrue(path.exists())
        target.write_text('[]')
        self.assertEqual(producer._finalize_deletions(), [])
        target.write_text(json.dumps(correct))
        target.with_name('duplicate.json').write_text(json.dumps(correct))
        self.assertEqual(len(producer._finalize_deletions()), 1)
        self.assertEqual(producer._finalize_deletions(), [])

    def test_upload_failure_persists_cleanup_and_receipt_for_restart_retry(self):
        host, _path, _producer, _catalog, request = self.setup_deletion()
        self.fail_upload = True
        self.monitor._sync_host(host)
        self.assertEqual(self.catalog.runs()['runs'], [])
        receipt = self.monitor.receipt_retry_root / host / (request['deletion_request_id'] + '.json')
        self.assertTrue(receipt.exists())
        replacement = instra_monitoring.MonitoringService(self.network, self.catalog,
            storage_root=self.monitor.storage_root, start_worker=False)
        self.addCleanup(replacement.close)
        self.catalog.monitoring = replacement
        self.assertEqual(replacement.paths(), ())
        self.fail_upload = False
        replacement._sync_host(host)
        self.assertEqual(len(self.uploads), 1)
        replacement._sync_host(host)
        self.assertEqual(len(self.uploads), 2)
        self.assertFalse(any(call[-1] for call in self.network.transfer_calls[5:]))

    def test_bad_notice_path_cannot_remove_unrelated_file_or_acknowledge(self):
        host, _path, _producer, catalog, request = self.setup_deletion()
        notice_path = catalog.root / '.instra_deletions/notices' / (request['deletion_request_id'] + '.json')
        notice = json.loads(notice_path.read_text())
        for bad in ('../charts.sqlite3', '/tmp/charts.sqlite3', 'run/../../charts.sqlite3', 'run\\charts.sqlite3'):
            with self.subTest(path=bad):
                notice_path.write_text(json.dumps({**notice, 'source_chart_path': bad}))
                self.monitor._sync_host(host)
                self.assertEqual(self.uploads, [])
                self.assertEqual(len(self.catalog.runs()['runs']), 1)

    def test_failed_cache_removal_does_not_send_receipt(self):
        host, _path, _producer, _catalog, _request = self.setup_deletion()
        with patch.object(self.monitor, '_delete_chart_files', side_effect=PermissionError('locked')):
            self.monitor._sync_host(host)
        self.assertEqual(self.uploads, [])
        self.monitor._sync_host(host)
        self.assertEqual(len(self.uploads), 1)

    def test_final_cleanup_failure_remains_pending_and_recovers(self):
        host, path, producer, _catalog, request = self.setup_deletion()
        self.monitor._sync_host(host)
        with patch.object(producer, '_delete_chart_files', side_effect=PermissionError('locked')):
            self.assertEqual(producer._finalize_deletions(), [])
        self.assertIn(request['deletion_request_id'], producer.deletions)
        self.assertTrue(path.exists())
        self.assertEqual(len(producer._finalize_deletions()), 1)

    def test_notice_transfer_failure_retries_without_acknowledging(self):
        host, _path, _producer, _catalog, _request = self.setup_deletion(acquired=False)
        transfer = self.network.monitor_transfer
        def fail_notice(host_id, root_kind, relative, destination):
            if relative.startswith('.instra_deletions/notices/'):
                raise instra_network.NetworkError('transfer', 'notice unavailable', host_id)
            return transfer(host_id, root_kind, relative, destination)
        with patch.object(self.network, 'monitor_transfer', side_effect=fail_notice):
            self.monitor._sync_host(host)
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.network.transfer_calls, [])
        self.assertEqual(self.catalog.runs()['runs'], [])
        self.monitor._sync_host(host)
        self.assertEqual(len(self.uploads), 1)

    def test_inflight_acquisition_finishes_before_receipt_and_cannot_restore_after_it(self):
        host, path = self._producer('producer')
        producer, catalog = self.producer_service(host, path)
        run_id = catalog.runs()['runs'][0]['dashboard_run_id']
        entered, release = threading.Event(), threading.Event()
        original = self.network.monitor_transfer
        def transfer(*args, **kwargs):
            if kwargs.get('database'):
                entered.set()
                self.assertTrue(release.wait(3))
            return original(*args, **kwargs)
        self.network.monitor_transfer = transfer
        thread = threading.Thread(target=self.monitor._sync_host, args=(host,))
        thread.start()
        self.assertTrue(entered.wait(2))
        request = producer.begin_authoritative_delete(run_id)
        self.uploads = []
        self.network.monitor_upload_receipt = lambda _host, _path, payload: self.uploads.append(payload)
        self.monitor._sync_host(host)
        self.assertEqual(self.uploads, [])
        release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.monitor._sync_host(host)
        self.assertEqual(len(self.uploads), 1)
        self.assertEqual(self.catalog.runs()['runs'], [])
        self.monitor._sync_host(host)
        self.assertEqual(self.catalog.runs()['runs'], [])
        self.assertEqual(self.uploads[-1]['deletion_request_id'], request['deletion_request_id'])

    def test_force_delete_is_background_work_and_excludes_until_explicit_resume(self):
        host, path = self._producer('producer')
        self.monitor._sync_host(host)
        run_id = self.catalog.runs()['runs'][0]['dashboard_run_id']
        entered, release = threading.Event(), threading.Event()
        original = self.monitor._remove_cached_remote_run
        def slow(*args):
            entered.set()
            self.assertTrue(release.wait(3))
            return original(*args)
        with patch.object(self.monitor, '_remove_cached_remote_run', side_effect=slow):
            before = time.monotonic()
            self.assertTrue(self.monitor.force_delete_local_copy(run_id)['queued'])
            self.assertLess(time.monotonic() - before, .3)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.monitor.paths(), ())
            release.set()
            self.wait_idle(host)
        self.assertEqual(self.catalog.runs()['runs'], [])
        self.assertTrue(path.exists())
        self.monitor._sync_host(host)
        self.assertEqual(self.catalog.runs()['runs'], [])
        self.assertEqual(self.monitor.force_delete_requests, {})
        excluded = self.monitor.deletion_snapshot()['excluded_local_copies']
        self.assertEqual(len(excluded), 1)
        self.monitor.resume_local_copy(host,excluded[0]['source_chart_path'])
        self.monitor._sync_host(host)
        self.assertEqual(len(self.catalog.runs()['runs']), 1)

    def test_force_delete_during_failed_acquisition_works_with_monitoring_disabled(self):
        host, _path = self._producer('producer')
        self.monitor._sync_host(host)
        run_id = self.catalog.runs()['runs'][0]['dashboard_run_id']
        self.network._host(host)['monitoring_enabled'] = False
        self.monitor.force_delete_local_copy(run_id)
        self.wait_idle(host)
        self.assertEqual(self.catalog.runs()['runs'], [])
        self.assertEqual(self.monitor.force_delete_requests, {})

    def test_force_exclusion_survives_restart_and_can_be_resumed(self):
        host, path = self._producer('producer')
        self.monitor._sync_host(host)
        run_id = self.catalog.runs()['runs'][0]['dashboard_run_id']
        self.monitor.force_delete_local_copy(run_id)
        self.wait_idle(host)
        replacement = instra_monitoring.MonitoringService(self.network,self.catalog,
            storage_root=self.monitor.storage_root,start_worker=False)
        self.addCleanup(replacement.close)
        self.catalog.monitoring = replacement
        for _ in range(3):
            replacement._sync_host(host)
            self.assertEqual(replacement.paths(), ())
        self.assertTrue(path.exists())
        item = replacement.deletion_snapshot()['excluded_local_copies'][0]
        self.assertEqual(item['run_id'], run_id)
        replacement.resume_local_copy(host,item['source_chart_path'])
        replacement._sync_host(host)
        self.assertEqual(len(replacement.paths()), 1)

    def test_old_producer_root_exclusion_does_not_delete_a_new_root_copy_on_restart(self):
        host, _path = self._producer('producer')
        self.monitor._sync_host(host)
        manifest = self.monitor.manifests[host]
        relative = next(iter(manifest['runs']))
        self.monitor.excluded_copies = {host: {relative: {'run_id': 'old_run', 'logs_root': '/retired-producer-logs'}}}
        instra_monitoring._atomic_json(self.monitor.excluded_copies_path, self.monitor.excluded_copies)
        replacement = instra_monitoring.MonitoringService(self.network, self.catalog,
            storage_root=self.monitor.storage_root, start_worker=False)
        self.addCleanup(replacement.close)
        self.assertEqual(replacement.force_delete_requests, {})
        self.assertEqual(len(replacement.paths()), 1)

    def test_grid_requests_each_member_and_retains_unrelated_run(self):
        host, path, producer, catalog, request = self.setup_deletion()
        second = path.parent.parent / 'second/charts.sqlite3'
        store = LocalChartStore(second, run_name='second', run_id='second', config={})
        store.close()
        third = path.parent.parent / 'unrelated/charts.sqlite3'
        store = LocalChartStore(third, run_name='unrelated', run_id='unrelated', config={})
        store.close()
        members = [run['dashboard_run_id'] for run in catalog.runs()['runs'] if run['local_run_id'] == 'second']
        result = catalog.delete_runs(members)
        self.assertEqual(len(result['deleted_run_ids']), 1)
        self.monitor._sync_host(host)
        self.assertEqual(len(self.uploads), 2)
        self.assertEqual(len(producer._finalize_deletions()), 2)
        self.assertFalse(path.exists())
        self.assertFalse(second.exists())
        self.assertTrue(third.exists())

    def test_shared_cached_training_log_is_retained_for_unaffected_run(self):
        host, path = self._producer('producer')
        second = path.parent.parent / 'second/charts.sqlite3'
        store = LocalChartStore(second, run_name='second', run_id='second', config={})
        store.close()
        self.monitor._sync_host(host)
        relative = 'run/same_id/charts.sqlite3'
        manifest = self.monitor.manifests[host]
        cached_log = self.monitor._host_root(host) / 'logs/run/train.log'
        self.assertTrue(cached_log.exists())
        self.monitor._remove_cached_remote_run(host, relative, manifest)
        self.assertTrue(cached_log.exists())
        self.assertIn('run/second/charts.sqlite3', manifest['runs'])

    def test_settings_persist_and_validate(self):
        self.assertEqual(self.monitor.deletion_timeout_days(), 7)
        for bad in (0, 366, 1.5, True, '7', None):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.monitor.set_deletion_timeout_days(bad)
        self.monitor.set_deletion_timeout_days(11)
        replacement = instra_monitoring.MonitoringService(self.network, self.catalog,
            storage_root=self.monitor.storage_root, start_worker=False)
        self.addCleanup(replacement.close)
        self.assertEqual(replacement.deletion_timeout_days(), 11)
# ^^^ THOG

# vvv THOG receipt uploads use the existing SSH stream and reject mismatched destinations before transfer
class ReceiptTransportTests(unittest.TestCase):
    def test_upload_is_atomic_root_scoped_and_has_no_node_agent_dependency(self):
        from types import SimpleNamespace
        service = object.__new__(instra_network.NetworkService)
        service.local_id = 'thog_host.viewer'
        request_id = 'a' * 32
        relative = f'.instra_deletions/receipts/{request_id}/thog_host_viewer.json'
        payload = {'producer_host_id': 'thog_host.producer', 'responding_host_id': service.local_id,
                   'deletion_request_id': request_id, 'deletion_processed': True}
        host = {'thog_host_id': 'thog_host.producer', 'address': 'localhost', 'ssh_port': 22, 'ssh_user': 'user'}
        with patch.object(service, '_monitor_source', return_value=(host, '/producer/logs/' + relative)) as source, \
                patch.object(instra_network, '_ssh_options', return_value=['-o', 'ControlPath=/existing/connection']), \
                patch.object(instra_network.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            result = service.monitor_upload_receipt('thog_host.producer', relative, payload)
            command = run.call_args.args[0]
            self.assertEqual(command[0], 'ssh')
            self.assertIn('ControlPath=/existing/connection', command)
            self.assertIn(f'/producer/logs/.instra_deletions/notices/{request_id}.json', command[-1])
            self.assertIn('mv -f', command[-1])
            self.assertEqual(json.loads(run.call_args.kwargs['input']), payload)
            self.assertEqual(result['uploaded'], relative)
            self.assertEqual(source.call_args.args, ('thog_host.producer', 'logs', relative))

    def test_bad_paths_or_identity_never_start_ssh(self):
        service = object.__new__(instra_network.NetworkService)
        service.local_id = 'thog_host.viewer'
        request_id = 'a' * 32
        relative = f'.instra_deletions/receipts/{request_id}/thog_host_viewer.json'
        payload = {'producer_host_id': 'thog_host.producer', 'responding_host_id': service.local_id,
                   'deletion_request_id': request_id, 'deletion_processed': True}
        with patch.object(service, '_monitor_source') as source:
            for bad in ('../notice.json', relative + '\n', relative.replace(request_id, 'wrong'),
                        relative.replace('thog_host_viewer', 'thog_host_other'), relative.replace('/receipts/', '/notices/')):
                with self.subTest(path=bad), self.assertRaises(instra_network.NetworkError):
                    service.monitor_upload_receipt('thog_host.producer', bad, payload)
            with self.assertRaises(instra_network.NetworkError):
                service.monitor_upload_receipt('thog_host.producer', relative, {**payload, 'deletion_processed': False})
            source.assert_not_called()
# ^^^ THOG
