import unittest
from typing import cast

import semfn


class Response:
    def __init__(self, answers):
        self.answers = answers

    def model_dump(self, *, mode):
        assert mode == "json"
        return {"answers": self.answers, "model": "jev-test"}


class Client:
    def __init__(self):
        self.calls = []

    def system_one(self, **request):
        self.calls.append(request)
        return Response({"question_0": {"noul": 0.9}})


class JevBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapts_the_sdk_response(self):
        client = Client()
        backend = semfn.JevBackend("jev-test", client=client)

        response = await backend.evaluate(
            {"text": "Production is down"},
            {"question_0": {"type": "noul", "instructions": "Is it urgent?"}},
        )

        self.assertEqual(response["answers"]["question_0"]["noul"], 0.9)
        self.assertEqual(client.calls[0]["model"], "jev-test")
        self.assertEqual(client.calls[0]["state"], {"text": "Production is down"})

    def test_configure_selects_backend_defaults(self):
        runtime = semfn.configure(backend="jev")
        self.assertIsInstance(runtime.backend, semfn.JevBackend)
        jev = cast(semfn.JevBackend, runtime.backend)
        self.assertEqual(jev.model, "jev-latest")

        runtime = semfn.configure(backend="laya")
        self.assertIsInstance(runtime.backend, semfn.LayaBackend)
        laya = cast(semfn.LayaBackend, runtime.backend)
        self.assertEqual(
            laya.model,
            "convaiinnovations/laya-multilingual",
        )
