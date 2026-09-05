package api

import (
	"context"
	"log/slog"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
	"nhooyr.io/websocket"
)

// challenge_skip_freshness_test.go — skipChallenge must mean "auto-pass", not
// "never pass". Before the refresh loop, challengeLoop returned before starting
// its ticker, so LastChallengeVerified was written exactly once (registration
// attestation) and the routing liveness gate derouted the provider
// registry.ChallengeFreshnessMaxAge() later: every request 429'd with
// no_provider, /v1/models/capacity went empty and the warm pool saw nothing —
// while the provider stayed connected, online and warm.

// TestSkippedChallengeRefreshIntervalClamps pins the cadence. The e2e testbed
// parks the challenge interval at an hour precisely because nothing is sent, so
// following the configured interval alone would restamp long after the gate had
// already closed.
func TestSkippedChallengeRefreshIntervalClamps(t *testing.T) {
	maxAge := registry.ChallengeFreshnessMaxAge()
	cases := []struct {
		name     string
		interval time.Duration
		want     time.Duration
	}{
		{"testbed hour is clamped", time.Hour, MaxSkippedChallengeRefreshInterval},
		{"default interval is kept", DefaultChallengeInterval, DefaultChallengeInterval},
		{"short interval is kept", 20 * time.Millisecond, 20 * time.Millisecond},
		{"zero falls back", 0, MaxSkippedChallengeRefreshInterval},
		{"negative falls back", -time.Second, MaxSkippedChallengeRefreshInterval},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := skippedChallengeRefreshInterval(tc.interval)
			if got != tc.want {
				t.Fatalf("skippedChallengeRefreshInterval(%v) = %v, want %v", tc.interval, got, tc.want)
			}
			if got > maxAge/2 {
				t.Fatalf("refresh %v exceeds half the %v freshness window", got, maxAge)
			}
		})
	}
}

// TestSkipChallengeKeepsProviderRoutablePastFreshnessWindow drives the loop
// with a stamp already aged past the gate window — the state a long-lived
// devstack reaches after ~16 idle minutes — and asserts the provider comes back
// into routing and the capacity feed on its own.
func TestSkipChallengeKeepsProviderRoutablePastFreshnessWindow(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
	st := store.NewMemory(store.Config{AdminKey: "test-key"})
	reg := registry.New(logger)
	srv := NewServer(reg, st, ServerConfig{}, logger)
	srv.SetChallengeInterval(20 * time.Millisecond)
	srv.SetSkipChallenge(true)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const model = "skip-challenge-model"
	conn := connectProvider(t, ctx, ts.URL, []protocol.ModelInfo{
		{ID: model, ModelType: "chat", Quantization: "4bit"},
	}, testPublicKeyB64())
	defer conn.Close(websocket.StatusNormalClosure, "")

	makeProviderRoutable(reg)
	ids := reg.ProviderIDs()
	if len(ids) != 1 {
		t.Fatalf("provider count = %d, want 1", len(ids))
	}
	p := reg.GetProvider(ids[0])
	if p == nil {
		t.Fatal("provider not found in registry")
	}
	if findRoutableProvider(reg, model) == nil {
		t.Fatal("baseline: provider should be routable right after registration")
	}

	// Simulate the elapsed uptime: with challenges skipped, registration was
	// the only writer of this stamp.
	stale := time.Now().Add(-registry.ChallengeFreshnessMaxAge() - time.Minute)
	p.SetLastChallengeVerified(stale)
	if findRoutableProvider(reg, model) != nil {
		t.Fatal("precondition: a stale challenge stamp must close the routing gate")
	}

	// The skip-challenge refresh loop restamps within one tick.
	waitForCondition(t, 5*time.Second, "challenge freshness to be restamped", func() bool {
		return p.GetLastChallengeVerified().After(stale)
	})

	if findRoutableProvider(reg, model) == nil {
		t.Fatal("provider should be routable again after the freshness refresh")
	}
	if len(reg.ModelCapacitySnapshot()) == 0 {
		t.Fatal("/v1/models/capacity feed should list the model again")
	}
}

// waitForCondition polls until cond holds or the deadline passes.
func waitForCondition(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out after %v waiting for %s", timeout, what)
}
