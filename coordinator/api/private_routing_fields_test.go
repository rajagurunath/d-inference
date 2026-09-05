package api

import "testing"

func TestStripProviderRoutingFieldsRemovesPrivateIdentity(t *testing.T) {
	parsed := map[string]any{
		"model":            "test-model",
		"provider_serial":  "PRIVATE-SERIAL-A",
		"provider_serials": []any{"PRIVATE-SERIAL-B"},
		// The lane selector is consumed by resolveRequestLane in the prelude and
		// must not travel on to a provider (docs/design/tidal-batch-lane.md §3.6).
		"service_tier": "batch",
	}

	if !stripProviderRoutingFields(parsed) {
		t.Fatal("stripProviderRoutingFields returned false")
	}
	if _, ok := parsed["provider_serial"]; ok {
		t.Fatal("provider_serial was not stripped")
	}
	if _, ok := parsed["provider_serials"]; ok {
		t.Fatal("provider_serials was not stripped")
	}
	if _, ok := parsed["service_tier"]; ok {
		t.Fatal("service_tier was not stripped")
	}
	if parsed["model"] != "test-model" {
		t.Fatalf("unrelated request fields changed: %v", parsed)
	}
}
