package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
)

// batchTestAlias is a public alias pointing at batchTestModel, so the tests
// below can tell the name a CONSUMER uses apart from the build id the
// coordinator dispatches on.
const batchTestAlias = "test-known-alias"

// withBatchAlias publishes batchTestAlias -> batchTestModel on the env's
// registry, which is what makes batchModelResolver return a build id that
// differs from the requested name.
func withBatchAlias(e *batchEnv) {
	e.srv.registry.SetModelAliases(map[string]registry.AliasTarget{
		batchTestAlias: {Desired: batchTestModel},
	})
}

// restrictedKey mints a key whose AllowedModels holds exactly the given names.
func restrictedKey(t *testing.T, st *store.MemoryStore, accountID string, allowed ...string) (string, *store.APIKey) {
	t.Helper()
	raw, rec, err := st.CreateAPIKey(accountID, store.APIKeyCreate{
		Name: "restricted", AllowedModels: allowed,
	})
	if err != nil {
		t.Fatalf("create api key: %v", err)
	}
	return raw, rec
}

// aliasJSONLLines builds a JSONL input file whose every line names `model`.
func aliasJSONLLines(model string, n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, `{"custom_id":"req-%d","method":"POST","url":"/v1/chat/completions","body":{"model":%q,"messages":[{"role":"user","content":%q}]}}`+"\n",
			i, model, batchTestPrompt)
	}
	return b.String()
}

// createAliasFileBatch uploads a JSONL file naming `model` and creates a batch
// from it, both under `key`.
func createAliasFileBatch(t *testing.T, e *batchEnv, model, key string) (int, map[string]any) {
	t.Helper()
	status, file := e.uploadMultipart(aliasJSONLLines(model, 2), "batch", "in.jsonl", key)
	if status != http.StatusOK {
		t.Fatalf("upload: status=%d body=%v", status, file)
	}
	return e.postJSON("/v1/batches", map[string]any{
		"input_file_id": file["id"], "endpoint": "/v1/chat/completions",
		"completion_window": "24h",
	}, key)
}

// TestBatchAllowListIsCheckedOnTheRequestedName is the S2 regression. The
// file-form parser rewrites every line's model to the concrete build id before
// anything is stored, so checking a key's AllowedModels against the model that
// reaches dispatch compared a coordinator-internal build id with the ALIAS a
// consumer put on their key — and refused every batch such a key submitted,
// while the identical online request was allowed. The allow-list is now checked
// on the requested name, at creation, exactly as online.
func TestBatchAllowListIsCheckedOnTheRequestedName(t *testing.T) {
	e := newBatchEnv(t)
	withBatchAlias(e)
	aliasKey, _ := restrictedKey(t, e.st, e.account, batchTestAlias)

	status, batch := createAliasFileBatch(t, e, batchTestAlias, aliasKey)
	if status != http.StatusOK {
		t.Fatalf("a key allow-listed for %q was refused its own batch: status=%d body=%v",
			batchTestAlias, status, batch)
	}

	// The stored items carry the RESOLVED build id, so dispatch never repeats
	// alias resolution hours later.
	batchID, _ := batch["id"].(string)
	items, err := e.st.ListItems(batchID)
	if err != nil || len(items) == 0 {
		t.Fatalf("ListItems: %d items, err=%v", len(items), err)
	}
	blobs := e.srv.BatchBlobs()
	raw, err := blobs.Open(items[0].BlobRef)
	if err != nil {
		t.Fatalf("open item blob: %v", err)
	}
	var stored struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatalf("stored item body: %v", err)
	}
	if stored.Model != batchTestModel {
		t.Fatalf("stored item model=%q, want the resolved build id %q", stored.Model, batchTestModel)
	}
}

// TestBatchAllowListDeniesAnAliasAKeyDoesNotHold is the other half: widening
// must not have happened. A key that holds only the BUILD id is refused a batch
// that asks for the alias — the same answer the online path gives, and the
// reason the allow-list check cannot simply be moved onto the resolved name.
func TestBatchAllowListDeniesAnAliasAKeyDoesNotHold(t *testing.T) {
	e := newBatchEnv(t)
	withBatchAlias(e)
	buildKey, _ := restrictedKey(t, e.st, e.account, batchTestModel)

	status, body := createAliasFileBatch(t, e, batchTestAlias, buildKey)
	if status != http.StatusForbidden {
		t.Fatalf("status=%d body=%v, want 403", status, body)
	}
	if code := errorCode(t, body); code != "model_not_allowed" {
		t.Fatalf("error.code=%q body=%v, want model_not_allowed", code, body)
	}

	// Control: the same key on the same fixture may batch the build id it does
	// hold, so the refusal above is the allow-list and not a broken resolver.
	status, ok := createAliasFileBatch(t, e, batchTestModel, buildKey)
	if status != http.StatusOK {
		t.Fatalf("control: status=%d body=%v, want 200", status, ok)
	}
}

// TestInlineBatchStoresTheResolvedBuild is the S3 regression. handleBatchCreate
// stored req.Model verbatim, and the dispatcher prefers b.Model — so an inline
// batch re-resolved its alias at dispatch, up to 24 hours later, and an alias
// that moved in between silently rerouted the rest of the batch onto a
// different build. The row now holds the build id; the wire object still echoes
// the alias the consumer typed.
func TestInlineBatchStoresTheResolvedBuild(t *testing.T) {
	e := newBatchEnv(t)
	withBatchAlias(e)

	status, obj := e.postJSON("/v1/batches", map[string]any{
		"model":    batchTestAlias,
		"endpoint": "/v1/chat/completions",
		"requests": []any{map[string]any{
			"custom_id": "a",
			"body":      map[string]any{"messages": []any{map[string]any{"role": "user", "content": batchTestPrompt}}},
		}},
	}, e.key)
	if status != http.StatusAccepted {
		t.Fatalf("status=%d body=%v, want 202", status, obj)
	}
	if got, _ := obj["model"].(string); got != batchTestAlias {
		t.Fatalf("batch object model=%q, want the requested alias %q", got, batchTestAlias)
	}

	batchID, _ := obj["id"].(string)
	stored, ok := e.st.GetBatch(e.account, batchID)
	if !ok {
		t.Fatalf("GetBatch(%q): not found", batchID)
	}
	if stored.Model != batchTestModel {
		t.Fatalf("batches.model=%q, want the resolved build id %q — the dispatcher would re-resolve the alias",
			stored.Model, batchTestModel)
	}
	if stored.RequestedModel != batchTestAlias {
		t.Fatalf("batches.requested_model=%q, want %q", stored.RequestedModel, batchTestAlias)
	}
}

// TestDispatchBatchItemHonoursAnAliasScopedKey closes the loop at DISPATCH: the
// item body and the batch row both carry the build id by then, so a key scoped
// to the alias must still be able to run it. The equivalence is confined to the
// coordinator-stamped batch lane (laneModelAllowed) and never widens a key —
// the requested name was already checked at creation.
func TestDispatchBatchItemHonoursAnAliasScopedKey(t *testing.T) {
	const model = "test-model"
	const alias = "test-model-alias"
	srv, reg, st, ctx, _, _ := batchFakeProviderCapturingBody(t, model,
		protocol.UsageInfo{PromptTokens: 3, CompletionTokens: 1}, "Hello alias")
	reg.SetModelAliases(map[string]registry.AliasTarget{alias: {Desired: model}})

	_, aliasKey, err := st.CreateAPIKey("admin", store.APIKeyCreate{
		Name: "alias-scoped", AllowedModels: []string{alias},
	})
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}

	body := []byte(`{"model":"` + model + `","messages":[{"role":"user","content":"hi"}]}`)
	outcome, err := srv.DispatchBatchItem(ctx, "admin", aliasKey.ID, model, body)
	if err != nil {
		t.Fatalf("DispatchBatchItem: %v", err)
	}
	if outcome.ErrCode != "" {
		t.Fatalf("ErrCode=%q body=%s — an alias-scoped key was refused its own batch item",
			outcome.ErrCode, outcome.ResponseBody)
	}

	// Control: a key scoped to an UNRELATED alias is still refused, so the
	// equivalence above resolves the allow-list rather than skipping it.
	reg.SetModelAliases(map[string]registry.AliasTarget{
		alias: {Desired: model}, "other-alias": {Desired: "some-other-build"},
	})
	_, otherKey, err := st.CreateAPIKey("admin", store.APIKeyCreate{
		Name: "other-alias-scoped", AllowedModels: []string{"other-alias"},
	})
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	outcome, err = srv.DispatchBatchItem(ctx, "admin", otherKey.ID, model, body)
	if err != nil {
		t.Fatalf("DispatchBatchItem (other alias): %v", err)
	}
	if outcome.ErrCode != batchRequestFailedCode {
		t.Fatalf("ErrCode=%q body=%s, want %q", outcome.ErrCode, outcome.ResponseBody, batchRequestFailedCode)
	}
	if code := responseErrorCode(outcome.ResponseBody); code != "model_not_allowed" {
		t.Fatalf("error.code=%q body=%s, want model_not_allowed", code, outcome.ResponseBody)
	}
}
