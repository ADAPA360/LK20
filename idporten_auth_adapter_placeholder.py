#!/usr/bin/env python3
"""
idporten_auth_adapter_placeholder.py
====================================
Simulated adapter for Norwegian ID-porten OIDC authentication.
Used for local development and integration testing of the LK20 platform.
"""

import json
import time
from typing import Dict, Any, Optional

class IDPortenMock:
    def __init__(self):
        self.mock_users = {
            "teacher_01": {
                "sub": "uuid-teacher-01",
                "name": "Ola Nordmann",
                "role": "teacher",
                "org_no": "999888777"
            },
            "admin_01": {
                "sub": "uuid-admin-01",
                "name": "Kari Nordmann",
                "role": "admin",
                "org_no": "111222333"
            }
        }

    def get_auth_url(self, redirect_uri: str) -> str:
        """
        Returns a mock login URL.
        """
        return f"https://mock.idporten.no/auth?client_id=lk20&redirect_uri={redirect_uri}"

    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Simulates the token exchange.
        """
        return {
            "access_token": "mock_access_token_" + str(int(time.time())),
            "id_token": "mock_id_token",
            "expires_in": 3600
        }

    def get_userinfo(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Returns mock userinfo based on the token.
        """
        # In mock mode, we just return the teacher_01 for any valid-looking token
        return self.mock_users["teacher_01"]

def get_idporten_adapter() -> IDPortenMock:
    return IDPortenMock()

if __name__ == "__main__":
    adapter = get_idporten_adapter()
    print(f"Mock Login URL: {adapter.get_auth_url('http://localhost:8000/callback')}")
