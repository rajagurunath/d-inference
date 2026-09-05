package testbed

import (
	"log/slog"
	"os"
	"strings"
	"testing"
)

// The dev loop's restart test needs three things to survive a Stop/Start:
// the database, the key that addresses its rows, and the seal key that opens
// the blobs. These tests cover the plumbing for all three without a live
// Postgres — they assert the resolution rules and that SuiteConfig carries the
// values through NewSuite, which is where every earlier attempt lost them
// (`--postgres` named EIGENINFERENCE_DATABASE_URL in its help and nothing in
// e2e/ read it).

// SuiteConfig.DatabaseURL must be the ONLY way a suite attaches to a database
// it does not own. deps.PostgresLifecycle.SetEnv exports an ephemeral
// instance's URL into EIGENINFERENCE_DATABASE_URL, so a package-level env
// fallback would make the second suite in a test binary attach to the first
// suite's already-removed database. This test fails if one is reintroduced.
func TestSuiteIgnoresTheEnvironmentDatabaseURL(t *testing.T) {
	t.Setenv("EIGENINFERENCE_DATABASE_URL", "postgres://leaked-from-a-previous-suite/db")

	s := NewSuite(SuiteConfig{UseMemoryStore: false})
	if s.Config.DatabaseURL != "" {
		t.Fatalf("DatabaseURL picked up from the environment: %q", s.Config.DatabaseURL)
	}
}

func TestResolveEncryptionMnemonicPrefersExplicitThenMnemonicThenPrefixed(t *testing.T) {
	t.Setenv("MNEMONIC", "")
	t.Setenv("EIGENINFERENCE_MNEMONIC", "prefixed words")

	if got := ResolveEncryptionMnemonic("explicit words"); got != "explicit words" {
		t.Fatalf("explicit value not preferred: %q", got)
	}
	if got := ResolveEncryptionMnemonic(""); got != "prefixed words" {
		t.Fatalf("EIGENINFERENCE_MNEMONIC fallback = %q", got)
	}

	t.Setenv("MNEMONIC", "bare words")
	if got := ResolveEncryptionMnemonic(""); got != "bare words" {
		t.Fatalf("MNEMONIC must win over EIGENINFERENCE_MNEMONIC, got %q", got)
	}
}

func TestResolveEncryptionMnemonicEmptyWithoutEnv(t *testing.T) {
	t.Setenv("MNEMONIC", "")
	t.Setenv("EIGENINFERENCE_MNEMONIC", "")

	if got := ResolveEncryptionMnemonic(""); got != "" {
		t.Fatalf("ResolveEncryptionMnemonic = %q, want \"\"", got)
	}
}

// DevRestartMnemonic must be a shape sealedblob.DeriveKey accepts (12 or 24
// words), otherwise the dev stack would fail to start rather than fall back.
func TestDevRestartMnemonicIsTwelveWords(t *testing.T) {
	n := len(strings.Fields(DevRestartMnemonic))
	if n != 12 {
		t.Fatalf("DevRestartMnemonic has %d words, want 12", n)
	}
}

// NewSuite normalises a config; the restart knobs must come out the other side
// untouched, since nothing else re-reads the flags after Start.
func TestNewSuiteCarriesRestartConfig(t *testing.T) {
	cfg := SuiteConfig{
		UseMemoryStore:     false,
		DatabaseURL:        "postgres://localhost:5432/dinf_devstack?sslmode=disable",
		APIKey:             "sk-db-deadbeef",
		EncryptionMnemonic: DevRestartMnemonic,
	}
	s := NewSuite(cfg)

	if s.Config.DatabaseURL != cfg.DatabaseURL {
		t.Fatalf("DatabaseURL = %q, want %q", s.Config.DatabaseURL, cfg.DatabaseURL)
	}
	if s.Config.APIKey != cfg.APIKey {
		t.Fatalf("APIKey = %q, want %q", s.Config.APIKey, cfg.APIKey)
	}
	if s.Config.EncryptionMnemonic != cfg.EncryptionMnemonic {
		t.Fatalf("EncryptionMnemonic = %q, want the dev mnemonic", s.Config.EncryptionMnemonic)
	}
	if s.Config.UseMemoryStore {
		t.Fatal("UseMemoryStore must stay false")
	}
}

// The default suite is unchanged by all of this: no URL, no key, no mnemonic.
func TestDefaultSuiteConfigHasNoRestartOverrides(t *testing.T) {
	cfg := DefaultSuiteConfig()
	if cfg.DatabaseURL != "" || cfg.APIKey != "" || cfg.EncryptionMnemonic != "" {
		t.Fatalf("DefaultSuiteConfig must leave the restart knobs empty: %+v", cfg)
	}
}

func TestRedactDatabaseURLDropsPassword(t *testing.T) {
	got := redactDatabaseURL("postgres://user:hunter2@127.0.0.1:5432/db?sslmode=disable")
	if got != "postgres://user@127.0.0.1:5432/db?sslmode=disable" {
		t.Fatalf("redactDatabaseURL = %q", got)
	}
	if got := redactDatabaseURL("postgres://localhost:5432/dinf_devstack?sslmode=disable"); got !=
		"postgres://localhost:5432/dinf_devstack?sslmode=disable" {
		t.Fatalf("password-less URL was altered: %q", got)
	}
}

// adoptConfiguredKey is the whole of the "same key after a restart" story, so
// both branches are covered against the memory store — no live Postgres
// needed, since both implementations share the CreateKeyForAccount /
// ValidateKeyFull contract.
func TestAdoptConfiguredKeyReusesAKeyTheStoreKnows(t *testing.T) {
	st := NewMemoryStore()
	issued, err := st.CreateKeyForAccount("testbed-user-0")
	if err != nil {
		t.Fatalf("CreateKeyForAccount: %v", err)
	}

	s := &Suite{
		Logger:  slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
		Config:  SuiteConfig{APIKey: issued},
		PgStore: st,
	}
	accountID := "testbed-user-0"
	if got := s.adoptConfiguredKey(&accountID); got != issued {
		t.Fatalf("adoptConfiguredKey = %q, want the issued key", got)
	}
	if accountID != "testbed-user-0" {
		t.Fatalf("accountID rewritten to %q", accountID)
	}
}

func TestAdoptConfiguredKeyFallsBackWhenTheStoreDoesNotKnowIt(t *testing.T) {
	s := &Suite{
		Logger:  slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
		Config:  SuiteConfig{APIKey: "sk-db-not-in-this-store"},
		PgStore: NewMemoryStore(),
	}
	accountID := "testbed-user-0"
	if got := s.adoptConfiguredKey(&accountID); got != "" {
		t.Fatalf("adoptConfiguredKey = %q, want \"\" so the caller mints", got)
	}
}

func TestAdoptConfiguredKeyIsANoOpWithoutAConfiguredKey(t *testing.T) {
	s := &Suite{
		Logger:  slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
		Config:  SuiteConfig{},
		PgStore: NewMemoryStore(),
	}
	accountID := "testbed-user-0"
	if got := s.adoptConfiguredKey(&accountID); got != "" {
		t.Fatalf("adoptConfiguredKey = %q, want \"\"", got)
	}
}
