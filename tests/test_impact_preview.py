"""POST /documents/{id}/impact-preview 变更影响预演集成测试（独立临时数据库）。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _create(client, content=None, doc_id="cfg"):
    body = {"id": doc_id}
    if content is not None:
        body["content"] = content
    r = client.post("/documents", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _patch(client, doc_id, base_revision, request_id, patch):
    r = client.patch(
        f"/documents/{doc_id}",
        json={
            "base_revision": base_revision,
            "request_id": request_id,
            "patch": patch,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _preview(client, doc_id, base_revision, patch, roots):
    return client.post(
        f"/documents/{doc_id}/impact-preview",
        json={
            "base_revision": base_revision,
            "patch": patch,
            "root_document_ids": roots,
        },
    )


# ---------------------------------------------------------------- 基本预演


def test_basic_preview_resolved_changes_against_current_world(client):
    _create(client, {"host": "db.internal", "port": 5432}, doc_id="base")
    _create(
        client,
        {"database": {"$ref": "doc:base@current#"}, "debug": False},
        doc_id="app",
    )

    r = _preview(client, "base", 1, {"port": 6543}, ["app"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document_id"] == "base"
    assert body["base_revision"] == 1
    assert body["current_revision"] == 1
    assert body["replayed"] is False
    assert body["changed"] is True
    # 候选内容一并给出，但它从未落库
    assert body["candidate"] == {"host": "db.internal", "port": 6543}

    (root,) = body["roots"]
    assert root["document_id"] == "app"
    assert root["revision"] == 1
    assert root["dependencies"] == [
        {"document_id": "base", "revision": 1, "pointer": ""}
    ]
    # 差异按现有 diff 语义：只有 /database/port 变化
    assert root["resolved_changes"] == [
        {
            "op": "replace",
            "path": "/database/port",
            "before": 5432,
            "after": 6543,
            "missing": [],
        }
    ]
    assert "error" not in root


def test_preview_with_target_as_root(client):
    _create(client, {"a": 1, "b": {"x": 1}}, doc_id="t")
    r = _preview(client, "t", 1, {"a": 2, "b": None}, ["t"])
    assert r.status_code == 200, r.text
    (root,) = r.json()["roots"]
    assert root["document_id"] == "t"
    assert root["revision"] == 1
    assert root["dependencies"] == []
    assert root["resolved_changes"] == [
        {"op": "replace", "path": "/a", "before": 1, "after": 2, "missing": []},
        {"op": "remove", "path": "/b", "before": {"x": 1}, "after": None,
         "missing": ["after"]},
    ]


def test_preview_noop_patch_reports_changed_false_and_empty_changes(client):
    _create(client, {"a": 1}, doc_id="t")
    r = _preview(client, "t", 1, {"a": 1}, ["t"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is False
    assert body["replayed"] is False
    assert body["roots"][0]["resolved_changes"] == []


def test_preview_expired_base_replay(client):
    _create(client, {"a": 1, "b": 2}, doc_id="t")
    _patch(client, "t", 1, "req-1", {"a": 100})  # 当前 rev2
    # 基于 rev1 只改 b，与服务端改的 a 不冲突 → 自动重放
    r = _preview(client, "t", 1, {"b": 200}, ["t"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["replayed"] is True
    assert body["changed"] is True
    assert body["candidate"] == {"a": 100, "b": 200}
    assert body["current_revision"] == 2


# ---------------------------------------------------------------- 多根与去重


def test_multiple_roots_results_in_input_order(client):
    _create(client, {"a": 1}, doc_id="t")
    _create(client, {"x": {"$ref": "doc:t@current#/a"}}, doc_id="r1")
    _create(client, {"y": {"z": {"$ref": "doc:t@current#/a"}}}, doc_id="r2")
    r = _preview(client, "t", 1, {"a": 9}, ["r2", "r1"])
    assert r.status_code == 200, r.text
    roots = r.json()["roots"]
    assert [item["document_id"] for item in roots] == ["r2", "r1"]
    assert roots[0]["resolved_changes"][0]["path"] == "/y/z"
    assert roots[1]["resolved_changes"][0]["path"] == "/x"


def test_duplicate_roots_deduped_by_first_occurrence(client):
    _create(client, {"a": 1}, doc_id="t")
    _create(client, {"x": 1}, doc_id="r")
    r = _preview(client, "t", 1, {"a": 2}, ["r", "t", "r", "r", "t"])
    assert r.status_code == 200, r.text
    assert [item["document_id"] for item in r.json()["roots"]] == ["r", "t"]


def test_single_root_resolution_failure_does_not_affect_others(client):
    _create(client, {"a": 1, "b": 2}, doc_id="t")
    _create(client, {"ok": {"$ref": "doc:t@current#/a"}}, doc_id="good")
    # bad 引用不存在的指针；ghost 文档不存在
    _create(client, {"bad": {"$ref": "doc:t@current#/missing"}}, doc_id="bad")
    _create(client, {"g": {"$ref": "doc:ghost@current#"}}, doc_id="ghostref")

    r = _preview(client, "t", 1, {"a": 10}, ["good", "bad", "ghostref"])
    assert r.status_code == 200, r.text
    roots = {item["document_id"]: item for item in r.json()["roots"]}

    assert roots["good"]["resolved_changes"] == [
        {"op": "replace", "path": "/ok", "before": 1, "after": 10, "missing": []}
    ]
    assert "error" not in roots["good"]

    bad_error = roots["bad"]["error"]
    assert bad_error["error"] == "pointer_not_found"
    assert bad_error["status_code"] == 422
    assert bad_error["chain"][0]["document_id"] == "bad"
    assert bad_error["chain"][-1]["document_id"] == "t"
    assert bad_error["chain"][-1]["pointer"] == "/missing"
    # 失败根仍固定其修订，但不给差异
    assert roots["bad"]["revision"] == 1
    assert roots["bad"]["resolved_changes"] == []

    ghost_error = roots["ghostref"]["error"]
    assert ghost_error["error"] == "document_not_found"
    assert ghost_error["status_code"] == 422


def test_root_document_not_found_recorded_per_root(client):
    _create(client, {"a": 1}, doc_id="t")
    r = _preview(client, "t", 1, {"a": 2}, ["nope"])
    assert r.status_code == 200, r.text
    (root,) = r.json()["roots"]
    assert root["document_id"] == "nope"
    assert root["revision"] is None
    assert root["error"]["error"] == "document_not_found"


def test_candidate_breaks_reference_recorded_as_root_error(client):
    # 当前世界可解析（/k 存在）；候选删除 k → 候选世界 pointer_not_found
    _create(client, {"k": 7}, doc_id="t")
    _create(client, {"v": {"$ref": "doc:t@current#/k"}}, doc_id="r")
    r = _preview(client, "t", 1, {"k": None}, ["r"])
    assert r.status_code == 200, r.text
    (root,) = r.json()["roots"]
    assert root["error"]["error"] == "pointer_not_found"
    assert root["revision"] == 1


def test_candidate_creates_reference_cycle_recorded_per_root(client):
    # t.s 初始为空对象；候选把它改成引用 r.s，而 r.s 又引用 t.s → 成环
    _create(client, {"s": {}}, doc_id="t")
    _create(client, {"s": {"$ref": "doc:t@current#/s"}}, doc_id="r")
    r = _preview(
        client,
        "t",
        1,
        {"s": {"$ref": "doc:r@current#/s"}},
        ["r"],
    )
    assert r.status_code == 200, r.text
    (root,) = r.json()["roots"]
    assert root["error"]["error"] == "reference_cycle"
    assert root["error"]["status_code"] == 409


# ---------------------------------------------------------------- 覆盖语义


def test_historical_pinned_revision_is_not_overlaid(client):
    # r 同时引用 t 的当前修订与 rev1：候选只影响 @current / 当前修订号，
    # 显式历史修订 rev1 保持原样。
    _create(client, {"v": "old"}, doc_id="t")
    _patch(client, "t", 1, "req-1", {"v": "new"})  # 当前 rev2
    _create(
        client,
        {
            "cur": {"$ref": "doc:t@current#/v"},
            "old": {"$ref": "doc:t@1#/v"},
        },
        doc_id="r",
    )
    r = _preview(client, "t", 2, {"v": "preview"}, ["r"])
    assert r.status_code == 200, r.text
    (root,) = r.json()["roots"]
    # @current 路径变化；@1 历史路径不变
    assert root["resolved_changes"] == [
        {
            "op": "replace",
            "path": "/cur",
            "before": "new",
            "after": "preview",
            "missing": [],
        }
    ]
    deps = {(d["document_id"], d["revision"], d["pointer"]) for d in root["dependencies"]}
    assert deps == {("t", 2, "/v"), ("t", 1, "/v")}


def test_explicit_current_revision_number_is_overlaid(client):
    _create(client, {"v": 1}, doc_id="t")
    _create(client, {"x": {"$ref": "doc:t@1#/v"}}, doc_id="r")
    r = _preview(client, "t", 1, {"v": 2}, ["r"])
    assert r.status_code == 200, r.text
    (root,) = r.json()["roots"]
    # 显式钉在“当前修订号 1”上，候选覆盖的正是该修订，因此可见变化
    assert root["resolved_changes"][0]["path"] == "/x"
    assert root["resolved_changes"][0]["after"] == 2


def test_unrelated_dependency_document_is_not_modified(client):
    _create(client, {"v": 1}, doc_id="t")
    _create(client, {"w": 2}, doc_id="other")
    _create(
        client,
        {
            "a": {"$ref": "doc:t@current#/v"},
            "b": {"$ref": "doc:other@current#/w"},
        },
        doc_id="r",
    )
    r = _preview(client, "t", 1, {"v": 10}, ["r"])
    assert r.status_code == 200, r.text
    (root,) = r.json()["roots"]
    assert [c["path"] for c in root["resolved_changes"]] == ["/a"]


# ---------------------------------------------------------------- 候选整体失败


def test_preview_merge_conflict_returns_structured_error(client):
    _create(client, {"a": 1}, doc_id="t")
    _patch(client, "t", 1, "req-1", {"a": 100})  # 服务端 rev2
    r = _preview(client, "t", 1, {"a": 999}, ["t"])
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "conflict"
    assert detail["document_id"] == "t"
    assert detail["base_revision"] == 1
    assert detail["current_revision"] == 2
    assert detail["conflicts"] == [
        {"path": "/a", "values": {"base": 1, "current": 100, "theirs": 999},
         "missing": []}
    ]
    # 预演错误体不含 request_id（根本不存在该概念）
    assert "request_id" not in detail


def test_preview_unknown_base_revision(client):
    _create(client, {"a": 1}, doc_id="t")
    r = _preview(client, "t", 99, {"a": 2}, ["t"])
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_base_revision"
    assert detail["base_revision"] == 99
    assert detail["current_revision"] == 1


def test_preview_contract_failure_returns_schema_error(client):
    _create(client, {"a": 1, "b": "x"}, doc_id="t")
    schema = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
    }
    r = client.put("/documents/t/schema", json={"schema": schema})
    assert r.status_code == 200, r.text
    r = _preview(client, "t", 1, {"a": "oops", "b": None}, ["t"])
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "schema_validation_failed"
    assert {e["path"] for e in detail["errors"]} == {"", "/a"}


def test_preview_target_not_found(client):
    r = _preview(client, "ghost", 1, {}, ["x"])
    assert r.status_code == 404, r.text
    assert r.json()["detail"]["error"] == "not_found"


# ---------------------------------------------------------------- 无副作用


def test_preview_creates_no_revision(client):
    _create(client, {"a": 1}, doc_id="t")
    assert _preview(client, "t", 1, {"a": 2}, ["t"]).status_code == 200
    # 修订历史仍只有 rev1
    r = client.get("/documents/t/revisions")
    assert [item["revision"] for item in r.json()["items"]] == [1]
    # 当前文档内容未变
    assert client.get("/documents/t").json()["content"] == {"a": 1}


def test_preview_does_not_consume_request_id(client):
    _create(client, {"a": 1}, doc_id="t")
    assert _preview(client, "t", 1, {"a": 2}, ["t"]).status_code == 200
    # 预演后第一次真实使用 req-after 必须是非幂等新写入
    r = _patch(client, "t", 1, "req-after", {"a": 2})
    assert r["idempotent"] is False
    assert r["revision"] == 2
    # 再来一次才是幂等重放
    r2 = client.patch(
        "/documents/t",
        json={"base_revision": 1, "request_id": "req-after", "patch": {"a": 2}},
    )
    assert r2.json()["idempotent"] is True


def test_preview_ignores_active_lease_and_does_not_change_it(client):
    _create(client, {"a": 1}, doc_id="t")
    r = client.post(
        "/documents/t/leases", json={"holder": "worker", "ttl_seconds": 300}
    )
    assert r.status_code == 201, r.text
    before = client.get("/documents/t/leases").json()
    # 不携带 lease_token 的预演照常成功（只读不受租约门禁）
    prev = _preview(client, "t", 1, {"a": 2}, ["t"])
    assert prev.status_code == 200, prev.text
    after = client.get("/documents/t/leases").json()
    # 租约 token 与过期时间均未被预演改动
    assert after["lease_token"] == before["lease_token"]
    assert after["expires_at"] == before["expires_at"]
    assert after["holder"] == "worker"
    # 无 token 的真实写入仍被租约拒绝（预演没有放行任何写入路径）
    denied = client.patch(
        "/documents/t",
        json={"base_revision": 1, "request_id": "req-x", "patch": {"a": 3}},
    )
    assert denied.status_code == 409
    assert denied.json()["detail"]["error"] == "lease_conflict"


def test_preview_readonly_even_though_write_endpoint_method_post(client):
    # 连续预演两次结果完全一致（无状态、无落库）
    _create(client, {"a": 1}, doc_id="t")
    first = _preview(client, "t", 1, {"a": 2}, ["t"])
    second = _preview(client, "t", 1, {"a": 2}, ["t"])
    assert first.json() == second.json()


# ---------------------------------------------------------------- 请求体校验


def test_preview_requires_one_to_twenty_roots(client):
    _create(client, {"a": 1}, doc_id="t")
    r = _preview(client, "t", 1, {}, [])
    assert r.status_code == 422
    r = _preview(client, "t", 1, {}, [f"r{i}" for i in range(21)])
    assert r.status_code == 422


def test_preview_rejects_invalid_root_id_chars(client):
    _create(client, {"a": 1}, doc_id="t")
    r = _preview(client, "t", 1, {}, ["bad/id"])
    assert r.status_code == 422


def test_preview_rejects_bad_base_revision(client):
    _create(client, {"a": 1}, doc_id="t")
    r = client.post(
        "/documents/t/impact-preview",
        json={"base_revision": 0, "patch": {}, "root_document_ids": ["t"]},
    )
    assert r.status_code == 422
