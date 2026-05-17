#!/usr/bin/env python3
"""
sdk_contract_test.py
====================
Contract testing for the LK20 Digital Twin SDK.
Ensures that the API surface remains stable for external consumers.
"""

import unittest
from lk20_main import LK20MainApp, LK20MainConfig

class TestSDKContract(unittest.TestCase):
    def setUp(self):
        self.config = LK20MainConfig.from_project_root(".")
        self.app = LK20MainApp(self.config)

    def test_app_interface_methods(self):
        """
        Verifies that all required public methods exist on LK20MainApp.
        """
        required_methods = [
            'status', 'health', 'init_project', 'login', 'logout', 'whoami',
            'create_network', 'verify', 'snapshot', 'inspect', 'search',
            'student_view', 'upload_curriculum', 'validate_upload',
            'attach_upload', 'list_uploads', 'inspect_upload', 'coverage',
            'gaps', 'sample_canonical', 'ingest_canonical', 'canonical_status',
            'report_gov_benefits', 'gov_inspect_system', 'report_teacher',
            'report_school', 'read_audit'
        ]
        for method in required_methods:
            self.assertTrue(hasattr(self.app, method), f"Missing method: {method}")
            self.assertTrue(callable(getattr(self.app, method)), f"Method not callable: {method}")

    def test_config_paths(self):
        """
        Ensures LK20MainConfig provides necessary paths.
        """
        paths = self.config.paths()
        required_keys = ['project_root', 'data_dir', 'current_network_path', 'audit_log_path']
        for key in required_keys:
            self.assertIn(key, paths)
            self.assertIsNotNone(paths[key])

if __name__ == "__main__":
    unittest.main()
