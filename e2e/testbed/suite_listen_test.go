package testbed

import "testing"

// TestSuiteListenAddrIsHonoured asserts that SuiteConfig.ListenAddr survives
// NewSuite and that netListen binds to that exact address instead of the
// ephemeral 127.0.0.1:0 default. devstack needs a fixed port so its printed
// base URL stays stable across restarts.
func TestSuiteListenAddrIsHonoured(t *testing.T) {
	cfg := SuiteConfig{UseMemoryStore: true, ListenAddr: "127.0.0.1:18099"}
	s := NewSuite(cfg)
	if s == nil {
		t.Fatal("no suite")
	}
	if s.Config.ListenAddr != "127.0.0.1:18099" {
		t.Fatalf("ListenAddr not preserved on SuiteConfig: %q", s.Config.ListenAddr)
	}

	l, err := netListen(s.Config.ListenAddr)
	if err != nil {
		t.Fatalf("netListen(%q): %v", s.Config.ListenAddr, err)
	}
	defer l.inner.Close()
	if l.port != 18099 {
		t.Fatalf("got port %d, want 18099", l.port)
	}
	if l.baseURL != "http://127.0.0.1:18099" {
		t.Fatalf("got base URL %q, want http://127.0.0.1:18099", l.baseURL)
	}
}

// TestNetListenDefaultsToEphemeralPort preserves the pre-existing behaviour:
// an empty address still binds an OS-assigned ephemeral port on 127.0.0.1.
func TestNetListenDefaultsToEphemeralPort(t *testing.T) {
	l, err := netListen("")
	if err != nil {
		t.Fatalf("netListen(\"\"): %v", err)
	}
	defer l.inner.Close()
	if l.port == 0 {
		t.Fatal("expected a bound ephemeral port, got 0")
	}
}
